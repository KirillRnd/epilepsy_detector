#!/usr/bin/env python3
"""Tune hysteresis postprocessing parameters on processed EEG recordings."""

from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import itertools
import json
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Iterable

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.postprocessing.hysteresis import (
    HysteresisParams,
    evaluate_segments,
    finalize_hysteresis_segments,
    hysteresis_raw_segments,
    postprocess_samples,
    precision_recall_f1,
)
from src.data_loading.input_normalization import (
    apply_input_normalization,
    input_normalization_from_config,
)


DEFAULT_ONSETS = "0.2,0.3,0.4,0.5,0.6,0.7"
DEFAULT_OFFSETS = "0.05,0.1,0.15,0.2,0.3,0.4"
DEFAULT_MIN_DURATIONS = "1.0,2.0,3.0"
DEFAULT_MIN_GAPS = "1.0,2.0,3.0"
DEFAULT_COLLARS = "0.0,0.2"


def parse_float_list(value: str) -> list[float]:
    items = [item.strip() for item in value.split(",") if item.strip()]
    if not items:
        raise argparse.ArgumentTypeError("expected a comma-separated float list")
    return [float(item) for item in items]


def parse_str_list(value: str | None) -> list[str] | None:
    if value is None:
        return None
    items = [item.strip() for item in value.split(",") if item.strip()]
    return items or None


def load_config(path: Path | None) -> dict:
    if path is None:
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def select_animals(args: argparse.Namespace, config: dict) -> list[str] | None:
    explicit = parse_str_list(args.animals)
    if explicit:
        return explicit

    split = args.split
    data_cfg = config.get("data", {})
    if split == "all":
        return None
    key = f"{split}_animals"
    animals = data_cfg.get(key)
    if animals:
        return list(animals)
    return None


def read_sampling_rate(session_dir: Path, default_sr: float) -> float:
    metadata_path = session_dir / "conversion_metadata.json"
    if not metadata_path.exists():
        return default_sr
    try:
        with metadata_path.open("r", encoding="utf-8") as handle:
            metadata = json.load(handle)
    except (json.JSONDecodeError, OSError):
        return default_sr
    for key in ("sampling_freq", "sampling_frequency", "target_sr", "sfreq"):
        if key in metadata:
            return float(metadata[key])
    return default_sr


def load_true_segments(segments_path: Path) -> list[tuple[int, int]]:
    segments: list[tuple[int, int]] = []
    with segments_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row.get("segment_type") != "seizure":
                continue
            start = int(float(row["start_sample"]))
            end = int(float(row["end_sample"]))
            if end > start:
                segments.append((start, end))
    return segments


def discover_sessions(
    data_dir: Path,
    animals: list[str] | None,
    sessions_filter: list[str] | None,
    default_sr: float,
) -> list[dict]:
    selected_animals = set(animals) if animals else None
    selected_sessions = set(sessions_filter) if sessions_filter else None
    sessions: list[dict] = []

    for animal_dir in sorted(path for path in data_dir.iterdir() if path.is_dir()):
        animal_id = animal_dir.name
        if selected_animals and animal_id not in selected_animals:
            continue
        for session_dir in sorted(path for path in animal_dir.iterdir() if path.is_dir()):
            session_id = session_dir.name
            session_key = f"{animal_id}/{session_id}"
            if selected_sessions and session_key not in selected_sessions:
                continue

            signal_path = session_dir / "processed_signals.npy"
            segments_path = session_dir / "segments_info.csv"
            if not signal_path.exists() or not segments_path.exists():
                continue

            signals = np.load(signal_path, mmap_mode="r")
            sessions.append(
                {
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "key": session_key,
                    "dir": session_dir,
                    "signal_path": signal_path,
                    "segments_path": segments_path,
                    "n_samples": int(signals.shape[1]),
                    "sr": read_sampling_rate(session_dir, default_sr),
                    "target": load_true_segments(segments_path),
                }
            )

    return sessions


def resolve_device(requested_device: str) -> str:
    import torch

    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested_device == "cuda" and not torch.cuda.is_available():
        print("[WARN] CUDA requested but unavailable; using CPU")
        return "cpu"
    return requested_device


def load_model(model_name: str, checkpoint_path: Path, device: str):
    import torch
    import src.modeling  # noqa: F401 - registers model classes
    from src.modeling.model_registry import get_model_class

    model_class = get_model_class(model_name)
    model = model_class()

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    state_dict = checkpoint.get("state_dict", checkpoint)
    cleaned = {
        key[len("model.") :] if key.startswith("model.") else key: value
        for key, value in state_dict.items()
    }
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing:
        print(f"[WARN] Missing checkpoint keys: {missing}")
    if unexpected:
        print(f"[WARN] Unexpected checkpoint keys: {unexpected}")

    model.to(device)
    model.eval()
    return model


def sliding_inference(
    model,
    signal: np.ndarray,
    window_length: int,
    step: int,
    batch_size: int,
    device: str,
) -> np.ndarray:
    import torch

    n_total = int(signal.shape[1])
    prob_sum = np.zeros(n_total, dtype=np.float32)
    count = np.zeros(n_total, dtype=np.float32)

    starts = list(range(0, n_total - window_length + 1, step))
    if not starts or starts[-1] + window_length < n_total:
        starts.append(max(0, n_total - window_length))

    chunks: list[np.ndarray] = []
    positions: list[int] = []

    def flush() -> None:
        nonlocal chunks, positions
        batch = torch.from_numpy(np.stack(chunks, axis=0)).float().to(device)
        with torch.no_grad():
            logits = model(batch)
            probs = torch.sigmoid(logits).detach().cpu().numpy()
        for row, start in zip(probs, positions):
            actual_len = min(len(row), n_total - start)
            prob_sum[start : start + actual_len] += row[:actual_len].astype(np.float32)
            count[start : start + actual_len] += 1.0
        chunks = []
        positions = []

    for start in starts:
        end = start + window_length
        chunk = np.asarray(signal[:, start:end], dtype=np.float32)
        cur_len = int(chunk.shape[1])
        if cur_len < window_length:
            chunk = np.pad(chunk, ((0, 0), (0, window_length - cur_len)))
        chunks.append(chunk)
        positions.append(int(start))
        if len(chunks) >= batch_size:
            flush()

    if chunks:
        flush()

    count[count == 0] = 1.0
    return prob_sum / count


def probability_path(cache_dir: Path, session: dict, input_normalization: str) -> Path:
    safe_session = session["session_id"].replace("/", "__").replace("\\", "__")
    return cache_dir / input_normalization / session["animal_id"] / f"{safe_session}_probs.npy"


def attach_probability_paths(
    sessions: list[dict],
    cache_dir: Path,
    input_normalization: str,
) -> None:
    for session in sessions:
        session["prob_path"] = probability_path(cache_dir, session, input_normalization)


def needs_probability_recompute(sessions: list[dict], recompute_probs: bool) -> bool:
    if recompute_probs:
        return True
    return any(not Path(session["prob_path"]).exists() for session in sessions)


def ensure_probabilities(
    sessions: list[dict],
    model,
    args: argparse.Namespace,
    verbose: bool = True,
) -> None:
    cache_dir = Path(args.prob_cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    for idx, session in enumerate(sessions, start=1):
        prob_path = Path(session["prob_path"])
        prob_path.parent.mkdir(parents=True, exist_ok=True)

        if prob_path.exists() and not args.recompute_probs:
            if verbose:
                print(f"[{idx}/{len(sessions)}] Reusing probabilities: {session['key']}")
            session["probs"] = np.load(prob_path, mmap_mode="r")
            continue

        if model is None:
            raise RuntimeError(
                f"Probability cache is missing for {session['key']}: {prob_path}"
            )

        if verbose:
            print(f"[{idx}/{len(sessions)}] Running model: {session['key']}")
        signal = np.load(session["signal_path"], mmap_mode="r")
        signal = apply_input_normalization(signal, args.input_normalization)
        probs = sliding_inference(
            model=model,
            signal=signal,
            window_length=args.window_length,
            step=args.step,
            batch_size=args.batch_size,
            device=args.device,
        )
        np.save(prob_path, probs.astype(np.float32))
        session["probs"] = np.load(prob_path, mmap_mode="r")


def make_param_grid(args: argparse.Namespace) -> list[HysteresisParams]:
    params: list[HysteresisParams] = []
    for onset, offset, min_duration, min_gap, collar in itertools.product(
        args.onsets,
        args.offsets,
        args.min_durations,
        args.min_gaps,
        args.collars,
    ):
        if offset >= onset:
            continue
        params.append(
            HysteresisParams(
                onset=onset,
                offset=offset,
                min_duration_s=min_duration,
                min_gap_s=min_gap,
                collar_s=collar,
            )
        )
    if not params:
        raise ValueError("empty parameter grid; every offset must be lower than onset")
    return params


def aggregate_metrics(
    sessions: Iterable[dict],
    params: HysteresisParams,
    event_iou_threshold: float,
    min_event_overlap_s: float,
) -> dict:
    sample_tp = 0
    sample_pred = 0
    sample_true = 0
    event_tp = 0
    event_pred = 0
    event_true = 0

    for session in sessions:
        sr = float(session["sr"])
        predicted = postprocess_samples(
            session["probs"],
            sr=sr,
            onset=params.onset,
            offset=params.offset,
            min_duration_s=params.min_duration_s,
            min_gap_s=params.min_gap_s,
            collar_s=params.collar_s,
        )
        metrics = evaluate_segments(
            predicted,
            session["target"],
            event_iou_threshold=event_iou_threshold,
            min_event_overlap_samples=max(1, int(min_event_overlap_s * sr)),
        )
        sample_tp += metrics.sample_tp
        sample_pred += metrics.sample_pred
        sample_true += metrics.sample_true
        event_tp += metrics.event_tp
        event_pred += metrics.event_pred
        event_true += metrics.event_true
        del predicted, metrics
        gc.collect()

    sample_precision, sample_recall, sample_f1 = precision_recall_f1(
        sample_tp, sample_pred, sample_true
    )
    event_precision, event_recall, event_f1 = precision_recall_f1(
        event_tp, event_pred, event_true
    )

    return {
        **asdict(params),
        "sample_precision": sample_precision,
        "sample_recall": sample_recall,
        "sample_f1": sample_f1,
        "event_precision": event_precision,
        "event_recall": event_recall,
        "event_f1": event_f1,
        "sample_tp": sample_tp,
        "sample_pred": sample_pred,
        "sample_true": sample_true,
        "event_tp": event_tp,
        "event_pred": event_pred,
        "event_true": event_true,
    }


def aggregate_threshold_metrics(
    sessions: Iterable[dict],
    params_list: list[HysteresisParams],
    event_iou_threshold: float,
    min_event_overlap_s: float,
) -> list[dict]:
    if not params_list:
        return []

    onset = params_list[0].onset
    offset = params_list[0].offset
    totals = [
        {
            "sample_tp": 0,
            "sample_pred": 0,
            "sample_true": 0,
            "event_tp": 0,
            "event_pred": 0,
            "event_true": 0,
        }
        for _ in params_list
    ]

    for session in sessions:
        sr = float(session["sr"])
        probs = session["probs"]
        raw_segments = hysteresis_raw_segments(
            probs,
            onset=onset,
            offset=offset,
        )
        n_samples = int(probs.shape[0])

        for idx, params in enumerate(params_list):
            predicted = finalize_hysteresis_segments(
                raw_segments,
                n_samples=n_samples,
                sr=sr,
                min_duration_s=params.min_duration_s,
                min_gap_s=params.min_gap_s,
                collar_s=params.collar_s,
            )
            metrics = evaluate_segments(
                predicted,
                session["target"],
                event_iou_threshold=event_iou_threshold,
                min_event_overlap_samples=max(1, int(min_event_overlap_s * sr)),
            )
            totals[idx]["sample_tp"] += metrics.sample_tp
            totals[idx]["sample_pred"] += metrics.sample_pred
            totals[idx]["sample_true"] += metrics.sample_true
            totals[idx]["event_tp"] += metrics.event_tp
            totals[idx]["event_pred"] += metrics.event_pred
            totals[idx]["event_true"] += metrics.event_true

        del raw_segments
        gc.collect()

    rows = []
    for params, total in zip(params_list, totals):
        sample_precision, sample_recall, sample_f1 = precision_recall_f1(
            total["sample_tp"],
            total["sample_pred"],
            total["sample_true"],
        )
        event_precision, event_recall, event_f1 = precision_recall_f1(
            total["event_tp"],
            total["event_pred"],
            total["event_true"],
        )
        rows.append(
            {
                **asdict(params),
                "sample_precision": sample_precision,
                "sample_recall": sample_recall,
                "sample_f1": sample_f1,
                "event_precision": event_precision,
                "event_recall": event_recall,
                "event_f1": event_f1,
                **total,
            }
        )

    return rows


def write_results(rows: list[dict], output_csv: Path, best_json: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    with best_json.open("w", encoding="utf-8") as handle:
        json.dump(rows[0], handle, indent=2)


def prepare_sessions(args: argparse.Namespace, verbose: bool = True) -> list[dict]:
    config = load_config(Path(args.config) if args.config else None)
    if args.input_normalization is None:
        args.input_normalization = input_normalization_from_config(config)
    animals = select_animals(args, config)
    sessions = discover_sessions(
        data_dir=Path(args.data_dir),
        animals=animals,
        sessions_filter=parse_str_list(args.sessions),
        default_sr=args.sr,
    )
    if not sessions:
        raise SystemExit("No sessions found for the selected data/split filters")

    if verbose:
        print(f"Selected sessions: {len(sessions)}")
        for session in sessions:
            print(
                f"  {session['key']}: {session['n_samples'] / session['sr']:.1f}s, "
                f"{len(session['target'])} events"
            )

    attach_probability_paths(sessions, Path(args.prob_cache_dir), args.input_normalization)
    model = None
    if needs_probability_recompute(sessions, args.recompute_probs):
        args.device = resolve_device(args.device)
        model = load_model(
            model_name=args.model_name,
            checkpoint_path=Path(args.checkpoint),
            device=args.device,
        )
    ensure_probabilities(sessions, model, args, verbose=verbose)
    return sessions


def build_worker_command(
    args: argparse.Namespace,
    params_list: list[HysteresisParams],
    output_json: Path,
) -> list[str]:
    command = [
        sys.executable,
        "-m",
        "src.postprocessing.tune_hysteresis",
        "--data-dir",
        args.data_dir,
        "--config",
        args.config,
        "--checkpoint",
        args.checkpoint,
        "--model-name",
        args.model_name,
        "--split",
        args.split,
        "--window-length",
        str(args.window_length),
        "--step",
        str(args.step),
        "--batch-size",
        str(args.batch_size),
        "--device",
        args.device,
        "--sr",
        str(args.sr),
        "--event-iou-threshold",
        str(args.event_iou_threshold),
        "--min-event-overlap-s",
        str(args.min_event_overlap_s),
        "--metric",
        args.metric,
        "--prob-cache-dir",
        args.prob_cache_dir,
        "--input-normalization",
        args.input_normalization,
        "--_worker-param-json",
        json.dumps([asdict(params) for params in params_list], separators=(",", ":")),
        "--_worker-output-json",
        str(output_json),
    ]
    if args.animals:
        command.extend(["--animals", args.animals])
    if args.sessions:
        command.extend(["--sessions", args.sessions])
    return command


def run_grid_isolated(
    args: argparse.Namespace,
    grid: list[HysteresisParams],
) -> list[dict]:
    output_dir = Path(args.output_csv).parent / "_grid_workers"
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    grouped: dict[tuple[float, float], list[tuple[int, HysteresisParams]]] = {}
    for idx, params in enumerate(grid, start=1):
        grouped.setdefault((params.onset, params.offset), []).append((idx, params))

    by_index: dict[int, dict] = {}
    group_items = list(grouped.items())
    for group_idx, ((onset, offset), items) in enumerate(group_items, start=1):
        params_list = [params for _, params in items]
        params_json = json.dumps(
            [asdict(params) for params in params_list],
            sort_keys=True,
            separators=(",", ":"),
        )
        params_hash = hashlib.sha1(params_json.encode("utf-8")).hexdigest()[:12]
        output_json = output_dir / f"threshold_{group_idx:03d}_{params_hash}.json"
        if output_json.exists():
            with output_json.open("r", encoding="utf-8") as handle:
                group_rows = json.load(handle)
        else:
            command = build_worker_command(args, params_list, output_json)
            result = None
            for attempt in range(1, args.worker_retries + 2):
                result = subprocess.run(
                    command,
                    cwd=PROJECT_ROOT,
                    text=True,
                    capture_output=True,
                )
                if result.returncode == 0:
                    break

                print(
                    f"[WARN] Worker failed on threshold group {group_idx}/{len(group_items)} "
                    f"onset={onset:g} offset={offset:g} "
                    f"attempt {attempt}/{args.worker_retries + 1}"
                )
                if result.stdout:
                    print(result.stdout)
                if result.stderr:
                    print(result.stderr)

            if result is None or result.returncode != 0:
                print(
                    f"[ERROR] Worker failed permanently on threshold group "
                    f"{group_idx}/{len(group_items)} onset={onset:g} offset={offset:g}"
                )
                raise SystemExit(result.returncode if result else 1)

            with output_json.open("r", encoding="utf-8") as handle:
                group_rows = json.load(handle)

        for (original_idx, params), row in zip(items, group_rows):
            by_index[original_idx] = row
        last_idx = items[-1][0]
        best_group_row = max(group_rows, key=lambda row: row[args.metric])
        print(
            f"[{last_idx}/{len(grid)}] onset={onset:g} offset={offset:g} "
            f"best_{args.metric}={best_group_row[args.metric]:.4f}"
        )

    for idx in range(1, len(grid) + 1):
        rows.append(by_index[idx])
    return rows


def run_worker(args: argparse.Namespace) -> None:
    sessions = prepare_sessions(args, verbose=False)
    raw_params = json.loads(args._worker_param_json)
    if isinstance(raw_params, dict):
        raw_params = [raw_params]
    params_list = [
        HysteresisParams(
            onset=float(item["onset"]),
            offset=float(item["offset"]),
            min_duration_s=float(item["min_duration_s"]),
            min_gap_s=float(item["min_gap_s"]),
            collar_s=float(item["collar_s"]),
        )
        for item in raw_params
    ]
    rows = aggregate_threshold_metrics(
        sessions=sessions,
        params_list=params_list,
        event_iou_threshold=args.event_iou_threshold,
        min_event_overlap_s=args.min_event_overlap_s,
    )
    for row in rows:
        row["input_normalization"] = args.input_normalization
    output_path = Path(args._worker_output_json)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = output_path.with_suffix(output_path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as handle:
        json.dump(rows, handle)
    tmp_path.replace(output_path)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Tune hysteresis postprocessing parameters on data/processed2."
    )
    parser.add_argument("--data-dir", default="data/processed2")
    parser.add_argument("--config", default="experiments/config_processed2.yaml")
    parser.add_argument("--checkpoint", default="checkpoints/RDSCBiGRUv2.ckpt")
    parser.add_argument("--model-name", default="RDSCBiGRUDetector")
    parser.add_argument("--split", choices=["train", "val", "test", "all"], default="val")
    parser.add_argument("--animals", default=None, help="Comma-separated animal ids override")
    parser.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated animal/session keys, for example Ati5y1/BL_22Mart",
    )
    parser.add_argument("--window-length", type=int, default=2000)
    parser.add_argument("--step", type=int, default=1000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cuda", "cpu"],
    )
    parser.add_argument("--sr", type=float, default=400.0)
    parser.add_argument("--onsets", type=parse_float_list, default=parse_float_list(DEFAULT_ONSETS))
    parser.add_argument("--offsets", type=parse_float_list, default=parse_float_list(DEFAULT_OFFSETS))
    parser.add_argument(
        "--min-durations",
        type=parse_float_list,
        default=parse_float_list(DEFAULT_MIN_DURATIONS),
    )
    parser.add_argument("--min-gaps", type=parse_float_list, default=parse_float_list(DEFAULT_MIN_GAPS))
    parser.add_argument("--collars", type=parse_float_list, default=parse_float_list(DEFAULT_COLLARS))
    parser.add_argument("--event-iou-threshold", type=float, default=0.0)
    parser.add_argument("--min-event-overlap-s", type=float, default=0.1)
    parser.add_argument("--metric", choices=["event_f1", "sample_f1"], default="event_f1")
    parser.add_argument(
        "--output-csv",
        default="experiments/postprocessing/hysteresis_tuning/results.csv",
    )
    parser.add_argument(
        "--best-json",
        default="experiments/postprocessing/hysteresis_tuning/best_params.json",
    )
    parser.add_argument(
        "--prob-cache-dir",
        default="experiments/postprocessing/hysteresis_tuning/probabilities",
    )
    parser.add_argument(
        "--input-normalization",
        default=None,
        choices=[
            "none",
            "per_record_joint_zscore",
            "per_record_per_channel_zscore",
            "per_record_joint_robust",
            "per_record_per_channel_robust",
        ],
        help="Full-record input normalization. Defaults to config input_normalization or none.",
    )
    parser.add_argument("--recompute-probs", action="store_true")
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=5,
        help="Retries for an isolated grid-point worker before aborting.",
    )
    parser.add_argument(
        "--no-isolate-grid",
        action="store_true",
        help="Run grid in the current process instead of one worker process per point.",
    )
    parser.add_argument("--_worker-param-json", default=None, help=argparse.SUPPRESS)
    parser.add_argument("--_worker-output-json", default=None, help=argparse.SUPPRESS)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args._worker_param_json:
        run_worker(args)
        return

    sessions = prepare_sessions(args, verbose=True)

    grid = make_param_grid(args)
    print(f"Grid size: {len(grid)}")

    if args.no_isolate_grid:
        rows: list[dict] = []
        for idx, params in enumerate(grid, start=1):
            try:
                row = aggregate_metrics(
                    sessions=sessions,
                    params=params,
                    event_iou_threshold=args.event_iou_threshold,
                    min_event_overlap_s=args.min_event_overlap_s,
                )
            except Exception as exc:
                print(f"[ERROR] Failed on grid item {idx}/{len(grid)}: {asdict(params)}")
                raise exc
            rows.append(row)
            gc.collect()
            if idx == 1 or idx % 25 == 0 or idx == len(grid):
                print(
                    f"[{idx}/{len(grid)}] onset={params.onset:g} offset={params.offset:g} "
                    f"event_f1={row['event_f1']:.4f} sample_f1={row['sample_f1']:.4f}"
                )
    else:
        rows = run_grid_isolated(args, grid)

    for row in rows:
        row["input_normalization"] = args.input_normalization

    rows.sort(
        key=lambda row: (
            row[args.metric],
            row["event_f1"],
            row["sample_f1"],
            row["event_recall"],
            row["sample_recall"],
        ),
        reverse=True,
    )
    write_results(rows, Path(args.output_csv), Path(args.best_json))

    best = rows[0]
    print("Best parameters:")
    print(json.dumps(best, indent=2))
    print(f"Saved full grid to: {args.output_csv}")
    print(f"Saved best params to: {args.best_json}")


if __name__ == "__main__":
    main()
