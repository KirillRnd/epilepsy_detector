#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Final test evaluation for the epilepsy detector.

The script evaluates the locked inference configuration on processed test
recordings and writes a reproducible Markdown report plus machine-readable
tables. It keeps frame/chunk metrics separate from structured event metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import torch
import yaml

try:
    from sklearn.metrics import average_precision_score, roc_auc_score
except Exception:  # pragma: no cover - optional exact per-recording metrics
    average_precision_score = None
    roc_auc_score = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.modeling  # noqa: F401 - registers model classes
from inference import load_model, sliding_inference
from src.postprocessing import postprocess_samples


EVENT_LEVELS: dict[str, dict[str, float]] = {
    "any": {
        "min_ref_coverage": 0.0,
        "min_pred_coverage": 0.0,
        "min_iou": 0.0,
    },
    "weak": {
        "min_ref_coverage": 0.25,
        "min_pred_coverage": 0.10,
        "min_iou": 0.10,
    },
    "standard": {
        "min_ref_coverage": 0.50,
        "min_pred_coverage": 0.25,
        "min_iou": 0.25,
    },
    "strict": {
        "min_ref_coverage": 0.80,
        "min_pred_coverage": 0.50,
        "min_iou": 0.50,
    },
}


@dataclass
class RecordingResult:
    recording_id: str
    animal_id: str
    session_id: str
    pharmacology_group: str
    duration_s: float
    sr: float
    n_samples: int
    gt_event_count: int
    pred_event_count: int
    gt_burden_s: float
    pred_burden_s: float
    frame_metrics: dict[str, Any]
    chunk_metrics: dict[str, Any]
    event_metrics: dict[str, dict[str, Any]]
    split_merge: dict[str, Any]


class BinnedCurveAccumulator:
    """Approximate global AUROC/AUPRC without storing every frame."""

    def __init__(self, n_bins: int = 1000) -> None:
        self.n_bins = int(n_bins)
        self.pos = np.zeros(self.n_bins, dtype=np.int64)
        self.neg = np.zeros(self.n_bins, dtype=np.int64)

    def update(self, y_true: np.ndarray, y_prob: np.ndarray) -> None:
        y_true = np.asarray(y_true).reshape(-1).astype(bool)
        y_prob = np.asarray(y_prob).reshape(-1)
        if y_true.size == 0:
            return
        bins = np.clip((y_prob * (self.n_bins - 1)).astype(np.int64), 0, self.n_bins - 1)
        self.pos += np.bincount(bins[y_true], minlength=self.n_bins)
        self.neg += np.bincount(bins[~y_true], minlength=self.n_bins)

    def compute(self) -> dict[str, float | None]:
        total_pos = int(self.pos.sum())
        total_neg = int(self.neg.sum())
        if total_pos == 0 or total_neg == 0:
            return {"auroc": None, "auprc": None}

        tp = np.cumsum(self.pos[::-1]).astype(np.float64)
        fp = np.cumsum(self.neg[::-1]).astype(np.float64)
        recall = tp / total_pos
        fpr = fp / total_neg
        precision = np.divide(tp, tp + fp, out=np.ones_like(tp), where=(tp + fp) > 0)

        auroc = float(np.trapz(recall, fpr))
        auprc = float(np.trapz(precision, recall))
        return {"auroc": auroc, "auprc": auprc}

    def to_jsonable(self) -> dict[str, list[int]]:
        return {
            "pos": self.pos.astype(int).tolist(),
            "neg": self.neg.astype(int).tolist(),
        }

    @classmethod
    def from_jsonable(cls, payload: dict[str, list[int]]) -> "BinnedCurveAccumulator":
        obj = cls(n_bins=len(payload["pos"]))
        obj.pos = np.asarray(payload["pos"], dtype=np.int64)
        obj.neg = np.asarray(payload["neg"], dtype=np.int64)
        return obj


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(path_like: str | Path, base: Path = PROJECT_ROOT) -> Path:
    path = Path(path_like)
    if path.is_absolute():
        return path
    return (base / path).resolve()


def safe_div(num: float, den: float) -> float:
    return float(num / den) if den else 0.0


def f1_score(precision: float, recall: float) -> float:
    return safe_div(2.0 * precision * recall, precision + recall)


def none_if_nan(value: float | None) -> float | None:
    if value is None:
        return None
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    return float(value)


def exact_curve_metrics(y_true: np.ndarray, y_prob: np.ndarray) -> dict[str, float | None]:
    y_true = np.asarray(y_true).reshape(-1).astype(np.uint8)
    y_prob = np.asarray(y_prob).reshape(-1)
    if len(np.unique(y_true)) < 2:
        return {"auroc": None, "auprc": None}
    if roc_auc_score is None or average_precision_score is None:
        return {"auroc": None, "auprc": None}
    return {
        "auroc": none_if_nan(float(roc_auc_score(y_true, y_prob))),
        "auprc": none_if_nan(float(average_precision_score(y_true, y_prob))),
    }


def binary_metrics_from_counts(tn: int, fp: int, fn: int, tp: int) -> dict[str, Any]:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    npv = safe_div(tn, tn + fn)
    f1 = f1_score(precision, recall)
    return {
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "specificity": specificity,
        "npv": npv,
        "f1": f1,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "confusion_matrix": [[int(tn), int(fp)], [int(fn), int(tp)]],
    }


def compute_frame_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, Any]:
    y_true_b = np.asarray(y_true).reshape(-1).astype(bool)
    y_pred_b = np.asarray(y_pred).reshape(-1).astype(bool)
    tn = int(np.count_nonzero(~y_true_b & ~y_pred_b))
    fp = int(np.count_nonzero(~y_true_b & y_pred_b))
    fn = int(np.count_nonzero(y_true_b & ~y_pred_b))
    tp = int(np.count_nonzero(y_true_b & y_pred_b))
    metrics = binary_metrics_from_counts(tn, fp, fn, tp)
    metrics.update(exact_curve_metrics(y_true_b, y_prob))
    metrics["gt_positive_rate"] = safe_div(tp + fn, y_true_b.size)
    metrics["pred_positive_rate"] = safe_div(tp + fp, y_true_b.size)
    metrics["n_samples"] = int(y_true_b.size)
    return metrics


def downsample_mask_to_1hz(mask: np.ndarray, sr: float, positive_fraction: float = 0.5) -> np.ndarray:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    samples_per_chunk = max(1, int(round(sr)))
    n_chunks = int(math.ceil(mask.size / samples_per_chunk))
    out = np.zeros(n_chunks, dtype=bool)
    for idx in range(n_chunks):
        start = idx * samples_per_chunk
        end = min(mask.size, start + samples_per_chunk)
        out[idx] = safe_div(int(np.count_nonzero(mask[start:end])), end - start) >= positive_fraction
    return out


def downsample_prob_to_1hz(probs: np.ndarray, sr: float) -> np.ndarray:
    probs = np.asarray(probs).reshape(-1)
    samples_per_chunk = max(1, int(round(sr)))
    n_chunks = int(math.ceil(probs.size / samples_per_chunk))
    out = np.zeros(n_chunks, dtype=np.float32)
    for idx in range(n_chunks):
        start = idx * samples_per_chunk
        end = min(probs.size, start + samples_per_chunk)
        out[idx] = float(np.mean(probs[start:end])) if end > start else 0.0
    return out


def mask_from_intervals(intervals: Iterable[tuple[int, int]], n_samples: int) -> np.ndarray:
    mask = np.zeros(int(n_samples), dtype=bool)
    for start, end in intervals:
        start = max(0, int(start))
        end = min(int(n_samples), int(end))
        if end > start:
            mask[start:end] = True
    return mask


def intervals_from_mask(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask).reshape(-1).astype(bool)
    if mask.size == 0:
        return []
    padded = np.concatenate(([False], mask, [False]))
    diff = np.diff(padded.astype(np.int8))
    starts = np.flatnonzero(diff == 1)
    ends = np.flatnonzero(diff == -1)
    return [(int(s), int(e)) for s, e in zip(starts, ends)]


def gt_intervals_from_segments(segments_df: pd.DataFrame) -> list[tuple[int, int]]:
    seizure_df = segments_df[segments_df["segment_type"] == "seizure"]
    intervals = []
    for row in seizure_df.itertuples(index=False):
        intervals.append((int(row.start_sample), int(row.end_sample)))
    intervals.sort()
    return intervals


def interval_pair_metrics(gt: tuple[int, int], pred: tuple[int, int]) -> dict[str, float]:
    gt_start, gt_end = int(gt[0]), int(gt[1])
    pred_start, pred_end = int(pred[0]), int(pred[1])
    overlap = int(max(0, min(gt_end, pred_end) - max(gt_start, pred_start)))
    gt_duration = int(max(0, gt_end - gt_start))
    pred_duration = int(max(0, pred_end - pred_start))
    union = int(gt_duration + pred_duration - overlap)
    return {
        "overlap": float(overlap),
        "ref_coverage": float(overlap / gt_duration) if gt_duration else 0.0,
        "pred_coverage": float(overlap / pred_duration) if pred_duration else 0.0,
        "iou": float(overlap / union) if union else 0.0,
        "onset_error_samples": float(pred_start - gt_start),
        "offset_error_samples": float(pred_end - gt_end),
    }


def eligible_pair(pair: dict[str, float], level_config: dict[str, float]) -> bool:
    return (
        pair["overlap"] > 0
        and pair["ref_coverage"] >= level_config["min_ref_coverage"]
        and pair["pred_coverage"] >= level_config["min_pred_coverage"]
        and pair["iou"] >= level_config["min_iou"]
    )


def match_events(
    gt_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
    level_config: dict[str, float],
) -> list[dict[str, Any]]:
    if not gt_events or not pred_events:
        return []

    candidates: list[dict[str, Any]] = []
    pred_starts = np.asarray([int(p[0]) for p in pred_events], dtype=np.int64)
    pred_ends = np.asarray([int(p[1]) for p in pred_events], dtype=np.int64)

    for gt_idx, gt in enumerate(gt_events):
        gt_start, gt_end = int(gt[0]), int(gt[1])
        left = int(np.searchsorted(pred_ends, gt_start, side="right"))
        right = int(np.searchsorted(pred_starts, gt_end, side="left"))
        if right <= left:
            continue
        for pred_idx in range(left, right):
            pred = pred_events[pred_idx]
            pair = interval_pair_metrics(gt, pred)
            if eligible_pair(pair, level_config):
                candidates.append({"gt_index": gt_idx, "pred_index": pred_idx, **pair})

    if not candidates:
        return []

    candidates.sort(key=lambda c: (c["iou"], c["overlap"]), reverse=True)
    used_gt: set[int] = set()
    used_pred: set[int] = set()
    chosen = []
    for c in candidates:
        gt_idx = int(c["gt_index"])
        pred_idx = int(c["pred_index"])
        if gt_idx in used_gt or pred_idx in used_pred:
            continue
        used_gt.add(gt_idx)
        used_pred.add(pred_idx)
        chosen.append(c)
    return chosen


def summarize_match_values(matches: list[dict[str, Any]], key: str, scale: float = 1.0) -> float | None:
    if not matches:
        return None
    return float(np.mean([float(m[key]) / scale for m in matches]))


def compute_event_metrics_for_level(
    gt_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
    level_name: str,
    level_config: dict[str, float],
    duration_s: float,
    sr: float,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    matches = match_events(gt_events, pred_events, level_config)
    tp = len(matches)
    fp = len(pred_events) - tp
    fn = len(gt_events) - tp
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = f1_score(precision, recall)
    for match in matches:
        match["level"] = level_name
    metrics = {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": None,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "f1": f1,
        "missed_events": int(fn),
        "missed_events_pct": safe_div(fn, len(gt_events)),
        "false_events_per_hour": safe_div(fp, duration_s / 3600.0),
        "event_confusion_matrix": [[None, int(fp)], [int(fn), int(tp)]],
        "mean_iou": summarize_match_values(matches, "iou"),
        "mean_ref_coverage": summarize_match_values(matches, "ref_coverage"),
        "mean_pred_coverage": summarize_match_values(matches, "pred_coverage"),
        "mean_abs_onset_error_s": (
            float(np.mean([abs(float(m["onset_error_samples"])) / sr for m in matches]))
            if matches
            else None
        ),
        "mean_abs_offset_error_s": (
            float(np.mean([abs(float(m["offset_error_samples"])) / sr for m in matches]))
            if matches
            else None
        ),
    }
    return metrics, matches


def compute_split_merge(
    gt_events: list[tuple[int, int]],
    pred_events: list[tuple[int, int]],
) -> dict[str, Any]:
    if not gt_events:
        return {
            "split_gt_events": 0,
            "split_extra_pred_fragments": 0,
            "merged_pred_events": 0,
            "merge_extra_gt_events": 0,
            "fully_missed_gt_events_any_overlap": 0,
            "non_overlapping_pred_events": int(len(pred_events)),
        }
    if not pred_events:
        return {
            "split_gt_events": 0,
            "split_extra_pred_fragments": 0,
            "merged_pred_events": 0,
            "merge_extra_gt_events": 0,
            "fully_missed_gt_events_any_overlap": int(len(gt_events)),
            "non_overlapping_pred_events": 0,
        }

    gt_starts = np.asarray([int(g[0]) for g in gt_events], dtype=np.int64)
    gt_ends = np.asarray([int(g[1]) for g in gt_events], dtype=np.int64)
    pred_starts = np.asarray([int(p[0]) for p in pred_events], dtype=np.int64)
    pred_ends = np.asarray([int(p[1]) for p in pred_events], dtype=np.int64)

    gt_overlap_counts = []
    for gt_start, gt_end in zip(gt_starts, gt_ends):
        left = int(np.searchsorted(pred_ends, int(gt_start), side="right"))
        right = int(np.searchsorted(pred_starts, int(gt_end), side="left"))
        gt_overlap_counts.append(max(0, right - left))

    pred_overlap_counts = []
    for pred_start, pred_end in zip(pred_starts, pred_ends):
        left = int(np.searchsorted(gt_ends, int(pred_start), side="right"))
        right = int(np.searchsorted(gt_starts, int(pred_end), side="left"))
        pred_overlap_counts.append(max(0, right - left))

    return {
        "split_gt_events": int(sum(c > 1 for c in gt_overlap_counts)),
        "split_extra_pred_fragments": int(sum(max(0, c - 1) for c in gt_overlap_counts)),
        "merged_pred_events": int(sum(c > 1 for c in pred_overlap_counts)),
        "merge_extra_gt_events": int(sum(max(0, c - 1) for c in pred_overlap_counts)),
        "fully_missed_gt_events_any_overlap": int(sum(c == 0 for c in gt_overlap_counts)),
        "non_overlapping_pred_events": int(sum(c == 0 for c in pred_overlap_counts)),
    }


def session_dirs_for_animals(data_dir: Path, animals: list[str]) -> list[Path]:
    session_dirs: list[Path] = []
    for animal in animals:
        animal_dir = data_dir / animal
        if not animal_dir.exists():
            raise FileNotFoundError(f"Animal directory not found: {animal_dir}")
        for session_dir in sorted(p for p in animal_dir.iterdir() if p.is_dir()):
            if (session_dir / "processed_signals.npy").exists() and (session_dir / "segments_info.csv").exists():
                session_dirs.append(session_dir)
    return session_dirs


def filter_session_dirs(session_dirs: list[Path], session_keys: list[str] | None) -> list[Path]:
    if not session_keys:
        return session_dirs
    selected = set(session_keys)
    filtered = [
        session_dir
        for session_dir in session_dirs
        if f"{session_dir.parent.name}/{session_dir.name}" in selected
        or f"{session_dir.parent.name}_{session_dir.name}" in selected
    ]
    missing = selected - {
        f"{session_dir.parent.name}/{session_dir.name}" for session_dir in filtered
    } - {
        f"{session_dir.parent.name}_{session_dir.name}" for session_dir in filtered
    }
    if missing:
        raise ValueError(f"Requested sessions were not found in selected split: {sorted(missing)}")
    return filtered


def load_sr(session_dir: Path, fallback_sr: float) -> float:
    metadata_path = session_dir / "conversion_metadata.json"
    if not metadata_path.exists():
        return float(fallback_sr)
    with metadata_path.open("r", encoding="utf-8") as f:
        meta = json.load(f)
    return float(meta.get("sampling_freq", fallback_sr))


def load_pharmacology_groups(data_dir: Path) -> dict[tuple[str, str], str]:
    summary_path = data_dir / "conversion_summary.csv"
    if not summary_path.exists():
        return {}
    summary = pd.read_csv(summary_path)
    if "pharmacology_group" not in summary.columns:
        return {}
    groups: dict[tuple[str, str], str] = {}
    for row in summary.itertuples(index=False):
        animal_id = str(row.animal_id)
        session_id = str(row.session_id)
        group = str(row.pharmacology_group)
        if group and group.lower() != "nan":
            groups[(animal_id, session_id)] = group
    return groups


def infer_pharmacology_group(animal_id: str, session_id: str, groups: dict[tuple[str, str], str]) -> str:
    if (animal_id, session_id) in groups:
        return groups[(animal_id, session_id)]
    old_processed_dir = PROJECT_ROOT / "data" / "processed" / animal_id
    if old_processed_dir.exists():
        return "no_drug"
    return "drug" if animal_id.startswith("FN") else "unknown"


def predict_recording(
    model: torch.nn.Module,
    signals: np.ndarray,
    cfg: dict[str, Any],
    device: str,
) -> np.ndarray:
    return sliding_inference(
        model=model,
        data=signals,
        window_length=int(cfg.get("window_length", 2000)),
        step=int(cfg.get("step", 1000)),
        device=device,
        batch_size=int(cfg.get("batch_size", 64)),
    )


def row_for_pred_event(
    recording_id: str,
    animal_id: str,
    session_id: str,
    pharmacology_group: str,
    event_idx: int,
    interval: tuple[int, int],
    probs: np.ndarray,
    sr: float,
) -> dict[str, Any]:
    start, end = interval
    event_probs = probs[start:end]
    duration_s = (end - start) / sr
    return {
        "recording_id": recording_id,
        "animal_id": animal_id,
        "session_id": session_id,
        "pharmacology_group": pharmacology_group,
        "event_index": int(event_idx),
        "start_sample": int(start),
        "end_sample": int(end),
        "start_time_s": start / sr,
        "end_time_s": end / sr,
        "duration_s": duration_s,
        "confidence_mean": float(np.mean(event_probs)) if event_probs.size else None,
        "confidence_max": float(np.max(event_probs)) if event_probs.size else None,
        "confidence_integral": float(np.sum(event_probs) / sr) if event_probs.size else 0.0,
    }


def evaluate_recording(
    session_dir: Path,
    model: torch.nn.Module,
    inference_cfg: dict[str, Any],
    device: str,
    save_probabilities: bool,
    probabilities_dir: Path,
    global_frame_curves: BinnedCurveAccumulator,
    global_chunk_curves: BinnedCurveAccumulator,
    pharmacology_groups: dict[tuple[str, str], str],
) -> tuple[RecordingResult, list[dict[str, Any]], list[dict[str, Any]]]:
    animal_id = session_dir.parent.name
    session_id = session_dir.name
    recording_id = f"{animal_id}_{session_id}"
    pharmacology_group = infer_pharmacology_group(animal_id, session_id, pharmacology_groups)
    sr = load_sr(session_dir, float(inference_cfg.get("target_sr", 400.0)))

    signals = np.load(session_dir / "processed_signals.npy", mmap_mode="r")
    n_samples = int(signals.shape[1])
    segments_df = pd.read_csv(session_dir / "segments_info.csv")
    gt_events = gt_intervals_from_segments(segments_df)
    gt_mask = mask_from_intervals(gt_events, n_samples)

    probs = predict_recording(
        model=model,
        signals=signals,
        cfg=inference_cfg,
        device=device,
    )
    if probs.shape[0] != n_samples:
        probs = probs[:n_samples]

    pred_events = postprocess_samples(
        probs=probs,
        sr=sr,
        onset=float(inference_cfg.get("onset_threshold", 0.3)),
        offset=float(inference_cfg.get("offset_threshold", 0.15)),
        min_duration_s=float(inference_cfg.get("min_duration_s", 3.0)),
        min_gap_s=float(inference_cfg.get("min_gap_s", 2.0)),
        collar_s=float(inference_cfg.get("collar_s", 0.0)),
    )
    pred_mask = mask_from_intervals(pred_events, n_samples)

    frame_metrics = compute_frame_metrics(gt_mask, probs, pred_mask)
    global_frame_curves.update(gt_mask, probs)

    gt_chunk = downsample_mask_to_1hz(gt_mask, sr)
    pred_chunk = downsample_mask_to_1hz(pred_mask, sr)
    prob_chunk = downsample_prob_to_1hz(probs, sr)
    chunk_metrics = compute_frame_metrics(gt_chunk, prob_chunk, pred_chunk)
    global_chunk_curves.update(gt_chunk, prob_chunk)

    duration_s = n_samples / sr
    event_metrics: dict[str, dict[str, Any]] = {}
    match_rows: list[dict[str, Any]] = []
    for level_name, level_config in EVENT_LEVELS.items():
        metrics, matches = compute_event_metrics_for_level(
            gt_events=gt_events,
            pred_events=pred_events,
            level_name=level_name,
            level_config=level_config,
            duration_s=duration_s,
            sr=sr,
        )
        event_metrics[level_name] = metrics
        for match in matches:
            gt_start, gt_end = gt_events[int(match["gt_index"])]
            pred_start, pred_end = pred_events[int(match["pred_index"])]
            match_rows.append(
                {
                    "recording_id": recording_id,
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "pharmacology_group": pharmacology_group,
                    "level": level_name,
                    "gt_index": int(match["gt_index"]),
                    "pred_index": int(match["pred_index"]),
                    "gt_start_s": gt_start / sr,
                    "gt_end_s": gt_end / sr,
                    "pred_start_s": pred_start / sr,
                    "pred_end_s": pred_end / sr,
                    "overlap_s": float(match["overlap"]) / sr,
                    "ref_coverage": float(match["ref_coverage"]),
                    "pred_coverage": float(match["pred_coverage"]),
                    "iou": float(match["iou"]),
                    "onset_error_s": float(match["onset_error_samples"]) / sr,
                    "offset_error_s": float(match["offset_error_samples"]) / sr,
                }
            )

    pred_event_rows = [
        row_for_pred_event(recording_id, animal_id, session_id, pharmacology_group, idx, interval, probs, sr)
        for idx, interval in enumerate(pred_events)
    ]

    gt_burden_s = float(sum(end - start for start, end in gt_events) / sr)
    pred_burden_s = float(sum(end - start for start, end in pred_events) / sr)
    split_merge = compute_split_merge(gt_events, pred_events)

    if save_probabilities:
        probabilities_dir.mkdir(parents=True, exist_ok=True)
        np.save(probabilities_dir / f"{recording_id}_probabilities.npy", probs.astype(np.float32))

    result = RecordingResult(
        recording_id=recording_id,
        animal_id=animal_id,
        session_id=session_id,
        pharmacology_group=pharmacology_group,
        duration_s=duration_s,
        sr=sr,
        n_samples=n_samples,
        gt_event_count=len(gt_events),
        pred_event_count=len(pred_events),
        gt_burden_s=gt_burden_s,
        pred_burden_s=pred_burden_s,
        frame_metrics=frame_metrics,
        chunk_metrics=chunk_metrics,
        event_metrics=event_metrics,
        split_merge=split_merge,
    )
    return result, pred_event_rows, match_rows


def merge_binary_metric_counts(results: list[RecordingResult], key: str) -> dict[str, Any]:
    tn = sum(int(r.__dict__[key]["tn"]) for r in results)
    fp = sum(int(r.__dict__[key]["fp"]) for r in results)
    fn = sum(int(r.__dict__[key]["fn"]) for r in results)
    tp = sum(int(r.__dict__[key]["tp"]) for r in results)
    return binary_metrics_from_counts(tn, fp, fn, tp)


def merge_event_metrics(results: list[RecordingResult], level_name: str) -> dict[str, Any]:
    tp = sum(int(r.event_metrics[level_name]["tp"]) for r in results)
    fp = sum(int(r.event_metrics[level_name]["fp"]) for r in results)
    fn = sum(int(r.event_metrics[level_name]["fn"]) for r in results)
    duration_s = sum(r.duration_s for r in results)
    gt_events = sum(r.gt_event_count for r in results)
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    return {
        "tp": int(tp),
        "fp": int(fp),
        "fn": int(fn),
        "tn": None,
        "precision": precision,
        "recall": recall,
        "sensitivity": recall,
        "f1": f1_score(precision, recall),
        "missed_events": int(fn),
        "missed_events_pct": safe_div(fn, gt_events),
        "false_events_per_hour": safe_div(fp, duration_s / 3600.0),
        "event_confusion_matrix": [[None, int(fp)], [int(fn), int(tp)]],
    }


def biological_summary(results: list[RecordingResult]) -> dict[str, Any]:
    duration_s = sum(r.duration_s for r in results)
    gt_burden_s = sum(r.gt_burden_s for r in results)
    pred_burden_s = sum(r.pred_burden_s for r in results)
    gt_count = sum(r.gt_event_count for r in results)
    pred_count = sum(r.pred_event_count for r in results)
    return {
        "duration_s": duration_s,
        "gt_event_count": int(gt_count),
        "pred_event_count": int(pred_count),
        "event_count_error": int(pred_count - gt_count),
        "gt_burden_s": gt_burden_s,
        "pred_burden_s": pred_burden_s,
        "gt_burden_fraction": safe_div(gt_burden_s, duration_s),
        "pred_burden_fraction": safe_div(pred_burden_s, duration_s),
        "burden_error_s": pred_burden_s - gt_burden_s,
        "absolute_burden_error_s": abs(pred_burden_s - gt_burden_s),
        "event_rate_per_hour_gt": safe_div(gt_count, duration_s / 3600.0),
        "event_rate_per_hour_pred": safe_div(pred_count, duration_s / 3600.0),
    }


def build_group_rows(results: list[RecordingResult]) -> list[dict[str, Any]]:
    rows = []
    for group_name in sorted(set(r.pharmacology_group for r in results)):
        group_results = [r for r in results if r.pharmacology_group == group_name]
        bio = biological_summary(group_results)
        frame = merge_binary_metric_counts(group_results, "frame_metrics")
        chunk = merge_binary_metric_counts(group_results, "chunk_metrics")
        event_standard = merge_event_metrics(group_results, "standard")
        event_strict = merge_event_metrics(group_results, "strict")
        event_any = merge_event_metrics(group_results, "any")
        rows.append(
            {
                "pharmacology_group": group_name,
                "n_recordings": int(len(group_results)),
                "duration_s": bio["duration_s"],
                "gt_event_count": bio["gt_event_count"],
                "pred_event_count": bio["pred_event_count"],
                "gt_burden_s": bio["gt_burden_s"],
                "pred_burden_s": bio["pred_burden_s"],
                "burden_error_s": bio["burden_error_s"],
                "frame_f1": frame["f1"],
                "chunk_1hz_f1": chunk["f1"],
                "event_f1_standard": event_standard["f1"],
                "event_recall_any": event_any["recall"],
                "event_f1_strict": event_strict["f1"],
                "false_events_per_hour_standard": event_standard["false_events_per_hour"],
                "missed_events_standard": event_standard["missed_events"],
            }
        )
    return rows


def format_float(value: Any, digits: int = 4) -> str:
    if value is None:
        return "N/A"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, float) or isinstance(value, np.floating):
        if math.isnan(float(value)) or math.isinf(float(value)):
            return "N/A"
        return f"{float(value):.{digits}f}"
    return str(value)


def markdown_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(format_float(v) for v in row) + " |")
    return "\n".join(lines)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def read_csv_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def result_from_dict(payload: dict[str, Any]) -> RecordingResult:
    return RecordingResult(
        recording_id=payload["recording_id"],
        animal_id=payload["animal_id"],
        session_id=payload["session_id"],
        pharmacology_group=payload.get("pharmacology_group", "unknown"),
        duration_s=float(payload["duration_s"]),
        sr=float(payload["sr"]),
        n_samples=int(payload["n_samples"]),
        gt_event_count=int(payload["gt_event_count"]),
        pred_event_count=int(payload["pred_event_count"]),
        gt_burden_s=float(payload["gt_burden_s"]),
        pred_burden_s=float(payload["pred_burden_s"]),
        frame_metrics=payload["frame_metrics"],
        chunk_metrics=payload["chunk_metrics"],
        event_metrics=payload["event_metrics"],
        split_merge=payload["split_merge"],
    )


def build_recording_rows(results: list[RecordingResult]) -> list[dict[str, Any]]:
    rows = []
    for r in results:
        standard = r.event_metrics["standard"]
        strict = r.event_metrics["strict"]
        rows.append(
            {
                "recording_id": r.recording_id,
                "animal_id": r.animal_id,
                "session_id": r.session_id,
                "pharmacology_group": r.pharmacology_group,
                "duration_s": r.duration_s,
                "gt_event_count": r.gt_event_count,
                "pred_event_count": r.pred_event_count,
                "gt_burden_s": r.gt_burden_s,
                "pred_burden_s": r.pred_burden_s,
                "frame_f1": r.frame_metrics["f1"],
                "frame_precision": r.frame_metrics["precision"],
                "frame_recall": r.frame_metrics["recall"],
                "chunk_1hz_f1": r.chunk_metrics["f1"],
                "event_f1_standard": standard["f1"],
                "event_recall_any": r.event_metrics["any"]["recall"],
                "event_f1_strict": strict["f1"],
                "false_events_per_hour_standard": standard["false_events_per_hour"],
                "missed_events_standard": standard["missed_events"],
                "split_gt_events": r.split_merge["split_gt_events"],
                "merged_pred_events": r.split_merge["merged_pred_events"],
            }
        )
    return rows


def build_animal_rows(results: list[RecordingResult]) -> list[dict[str, Any]]:
    rows = []
    for animal_id, group in sorted(pd.DataFrame(build_recording_rows(results)).groupby("animal_id")):
        groups = sorted(set(str(v) for v in group["pharmacology_group"]))
        duration_s = float(group["duration_s"].sum())
        gt_count = int(group["gt_event_count"].sum())
        pred_count = int(group["pred_event_count"].sum())
        gt_burden_s = float(group["gt_burden_s"].sum())
        pred_burden_s = float(group["pred_burden_s"].sum())
        rows.append(
            {
                "animal_id": animal_id,
                "pharmacology_group": ",".join(groups),
                "n_recordings": int(len(group)),
                "duration_s": duration_s,
                "gt_event_count": gt_count,
                "pred_event_count": pred_count,
                "gt_burden_s": gt_burden_s,
                "pred_burden_s": pred_burden_s,
                "burden_error_s": pred_burden_s - gt_burden_s,
                "event_rate_per_hour_gt": safe_div(gt_count, duration_s / 3600.0),
                "event_rate_per_hour_pred": safe_div(pred_count, duration_s / 3600.0),
            }
        )
    return rows


def write_report(
    out_path: Path,
    args: argparse.Namespace,
    inference_cfg: dict[str, Any],
    train_cfg: dict[str, Any],
    results: list[RecordingResult],
    global_frame: dict[str, Any],
    global_chunk: dict[str, Any],
    global_event: dict[str, dict[str, Any]],
    bio: dict[str, Any],
    recording_rows: list[dict[str, Any]],
    animal_rows: list[dict[str, Any]],
    group_rows: list[dict[str, Any]],
) -> None:
    post_cfg_keys = [
        "onset_threshold",
        "offset_threshold",
        "min_duration_s",
        "min_gap_s",
        "collar_s",
    ]
    post_cfg = {k: inference_cfg.get(k) for k in post_cfg_keys}
    split_animals = train_cfg["data"].get(f"{args.split}_animals", [])
    now = datetime.now().isoformat(timespec="seconds")

    lines = [
        "# Финальный отчёт по evaluation",
        "",
        f"Сформирован: `{now}`",
        "",
        "## Входные данные",
        "",
        markdown_table(
            ["Item", "Value"],
            [
                ["data_dir", str(args.data_dir)],
                ["split", args.split],
                [f"{args.split}_animals", ", ".join(split_animals)],
                ["model_name", inference_cfg.get("model_name")],
                ["checkpoint", inference_cfg.get("checkpoint")],
                ["train_config", str(args.train_config)],
                ["inference_config", str(args.inference_config)],
                ["device", args.device],
            ],
        ),
        "",
        "## Postprocessing",
        "",
        markdown_table(["Parameter", "Value"], [[k, v] for k, v in post_cfg.items()]),
        "",
        "## Event Matching Contract v1",
        "",
        "Для event-level `TN` указан как `N/A`: для интервальной детекции нет естественного числа отрицательных событий. Поэтому event confusion matrix записана как `[[TN=N/A, FP], [FN, TP]]`.",
        "",
        markdown_table(
            ["Level", "ref coverage", "pred coverage", "IoU", "Смысл"],
            [
                ["any", "> 0 overlap", "> 0 overlap", "> 0 overlap", "событие хотя бы частично задетектировано"],
                ["weak", ">= 25%", ">= 10%", ">= 10%", "существенное частичное совпадение"],
                ["standard", ">= 50%", ">= 25%", ">= 25%", "основная event-метрика"],
                ["strict", ">= 80%", ">= 50%", ">= 50%", "событие хорошо восстановлено"],
            ],
        ),
        "",
        "## Глобальные frame metrics",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["precision", global_frame["precision"]],
                ["recall/sensitivity", global_frame["recall"]],
                ["specificity", global_frame["specificity"]],
                ["f1", global_frame["f1"]],
                ["balanced_accuracy", global_frame["balanced_accuracy"]],
                ["auroc_binned", global_frame.get("auroc")],
                ["auprc_binned", global_frame.get("auprc")],
                ["gt_positive_rate", global_frame.get("gt_positive_rate")],
                ["pred_positive_rate", global_frame.get("pred_positive_rate")],
            ],
        ),
        "",
        "Frame-level confusion matrix:",
        "",
        markdown_table(
            ["", "Pred 0", "Pred 1"],
            [
                ["GT 0", global_frame["tn"], global_frame["fp"]],
                ["GT 1", global_frame["fn"], global_frame["tp"]],
            ],
        ),
        "",
        "## Глобальные 1 Hz chunk metrics",
        "",
        markdown_table(
            ["Metric", "Value"],
            [
                ["precision", global_chunk["precision"]],
                ["recall/sensitivity", global_chunk["recall"]],
                ["specificity", global_chunk["specificity"]],
                ["f1", global_chunk["f1"]],
                ["balanced_accuracy", global_chunk["balanced_accuracy"]],
                ["auroc_binned", global_chunk.get("auroc")],
                ["auprc_binned", global_chunk.get("auprc")],
            ],
        ),
        "",
        "Chunk-level confusion matrix:",
        "",
        markdown_table(
            ["", "Pred 0", "Pred 1"],
            [
                ["GT 0", global_chunk["tn"], global_chunk["fp"]],
                ["GT 1", global_chunk["fn"], global_chunk["tp"]],
            ],
        ),
        "",
        "## Глобальные event metrics",
        "",
        markdown_table(
            ["Level", "TP", "FP", "FN", "TN", "Precision", "Recall", "F1", "False events/hour", "Missed %"],
            [
                [
                    level,
                    m["tp"],
                    m["fp"],
                    m["fn"],
                    "N/A",
                    m["precision"],
                    m["recall"],
                    m["f1"],
                    m["false_events_per_hour"],
                    m["missed_events_pct"],
                ]
                for level, m in global_event.items()
            ],
        ),
        "",
        "Event-level confusion matrices:",
        "",
    ]

    for level, m in global_event.items():
        lines.extend(
            [
                f"### {level}",
                "",
                markdown_table(
                    ["", "Pred event 0", "Pred event 1"],
                    [
                        ["GT event 0", "N/A", m["fp"]],
                        ["GT event 1", m["fn"], m["tp"]],
                    ],
                ),
                "",
            ]
        )

    lines.extend(
        [
            "## Биологическая сводка",
            "",
            markdown_table(
                ["Metric", "Value"],
                [
                    ["duration_h", bio["duration_s"] / 3600.0],
                    ["gt_event_count", bio["gt_event_count"]],
                    ["pred_event_count", bio["pred_event_count"]],
                    ["event_count_error", bio["event_count_error"]],
                    ["gt_burden_s", bio["gt_burden_s"]],
                    ["pred_burden_s", bio["pred_burden_s"]],
                    ["burden_error_s", bio["burden_error_s"]],
                    ["absolute_burden_error_s", bio["absolute_burden_error_s"]],
                    ["gt_burden_fraction", bio["gt_burden_fraction"]],
                    ["pred_burden_fraction", bio["pred_burden_fraction"]],
                    ["gt_event_rate_per_hour", bio["event_rate_per_hour_gt"]],
                    ["pred_event_rate_per_hour", bio["event_rate_per_hour_pred"]],
                ],
            ),
            "",
            "## Сводка по записям",
            "",
            markdown_table(
                [
                    "Recording",
                    "Group",
                    "GT events",
                    "Pred events",
                    "Frame F1",
                    "Chunk F1",
                    "Event F1 standard",
                    "Event recall any",
                    "Strict F1",
                    "FP/h standard",
                ],
                [
                    [
                        r["recording_id"],
                        r["pharmacology_group"],
                        r["gt_event_count"],
                        r["pred_event_count"],
                        r["frame_f1"],
                        r["chunk_1hz_f1"],
                        r["event_f1_standard"],
                        r["event_recall_any"],
                        r["event_f1_strict"],
                        r["false_events_per_hour_standard"],
                    ]
                    for r in recording_rows
                ],
            ),
            "",
            "## Сводка по pharmacology_group",
            "",
            markdown_table(
                [
                    "Group",
                    "Recordings",
                    "Duration h",
                    "GT events",
                    "Pred events",
                    "Frame F1",
                    "Event F1 standard",
                    "Event recall any",
                    "Strict F1",
                    "Burden error s",
                ],
                [
                    [
                        r["pharmacology_group"],
                        r["n_recordings"],
                        r["duration_s"] / 3600.0,
                        r["gt_event_count"],
                        r["pred_event_count"],
                        r["frame_f1"],
                        r["event_f1_standard"],
                        r["event_recall_any"],
                        r["event_f1_strict"],
                        r["burden_error_s"],
                    ]
                    for r in group_rows
                ],
            ),
            "",
            "## Сводка по животным",
            "",
            markdown_table(
                [
                    "Animal",
                    "Group",
                    "Recordings",
                    "Duration h",
                    "GT events",
                    "Pred events",
                    "GT burden s",
                    "Pred burden s",
                    "Burden error s",
                ],
                [
                    [
                        r["animal_id"],
                        r["pharmacology_group"],
                        r["n_recordings"],
                        r["duration_s"] / 3600.0,
                        r["gt_event_count"],
                        r["pred_event_count"],
                        r["gt_burden_s"],
                        r["pred_burden_s"],
                        r["burden_error_s"],
                    ]
                    for r in animal_rows
                ],
            ),
            "",
            "## Выходные файлы",
            "",
            markdown_table(
                ["File", "Содержимое"],
                [
                    ["report.md", "человекочитаемый отчёт"],
                    ["metrics.json", "полные machine-readable metrics"],
                    ["recording_metrics.csv", "сводка по записям"],
                    ["animal_summary.csv", "биологическая сводка по животным"],
                    ["pharmacology_group_summary.csv", "сводка по группам drug/no_drug"],
                    ["predicted_events.csv", "структурированные predicted events"],
                    ["event_matches.csv", "one-to-one event matches для каждого level"],
                ],
            ),
            "",
        ]
    )
    out_path.write_text("\n".join(lines), encoding="utf-8")


def write_final_outputs(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
    animals: list[str],
    results: list[RecordingResult],
    predicted_event_rows: list[dict[str, Any]],
    match_rows: list[dict[str, Any]],
    global_frame_curves: BinnedCurveAccumulator,
    global_chunk_curves: BinnedCurveAccumulator,
) -> None:
    global_frame = merge_binary_metric_counts(results, "frame_metrics")
    global_frame.update(global_frame_curves.compute())
    global_frame["gt_positive_rate"] = safe_div(
        global_frame["tp"] + global_frame["fn"], sum(r.n_samples for r in results)
    )
    global_frame["pred_positive_rate"] = safe_div(
        global_frame["tp"] + global_frame["fp"], sum(r.n_samples for r in results)
    )

    global_chunk = merge_binary_metric_counts(results, "chunk_metrics")
    global_chunk.update(global_chunk_curves.compute())
    global_event = {level: merge_event_metrics(results, level) for level in EVENT_LEVELS}
    bio = biological_summary(results)

    recording_rows = build_recording_rows(results)
    animal_rows = build_animal_rows(results)
    group_rows = build_group_rows(results)

    metrics = {
        "inputs": {
            "data_dir": str(args.data_dir),
            "train_config": str(args.train_config),
            "inference_config": str(args.inference_config),
            "split": args.split,
            "animals": animals,
            "model_name": inference_cfg.get("model_name"),
            "checkpoint": inference_cfg.get("checkpoint"),
            "device": args.device,
        },
        "event_matching_contract": EVENT_LEVELS,
        "postprocessing": {
            "onset_threshold": inference_cfg.get("onset_threshold"),
            "offset_threshold": inference_cfg.get("offset_threshold"),
            "min_duration_s": inference_cfg.get("min_duration_s"),
            "min_gap_s": inference_cfg.get("min_gap_s"),
            "collar_s": inference_cfg.get("collar_s"),
        },
        "frame_metrics": global_frame,
        "chunk_1hz_metrics": global_chunk,
        "event_metrics": global_event,
        "biological_summary": bio,
        "pharmacology_group_summary": group_rows,
        "recordings": [asdict(r) for r in results],
        "curve_bins": {
            "frame": global_frame_curves.to_jsonable(),
            "chunk_1hz": global_chunk_curves.to_jsonable(),
        },
    }

    (args.out / "metrics.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    write_csv(args.out / "recording_metrics.csv", recording_rows)
    write_csv(args.out / "animal_summary.csv", animal_rows)
    write_csv(args.out / "pharmacology_group_summary.csv", group_rows)
    write_csv(args.out / "predicted_events.csv", predicted_event_rows)
    write_csv(args.out / "event_matches.csv", match_rows)
    write_report(
        out_path=args.out / "report.md",
        args=args,
        inference_cfg=inference_cfg,
        train_cfg=train_cfg,
        results=results,
        global_frame=global_frame,
        global_chunk=global_chunk,
        global_event=global_event,
        bio=bio,
        recording_rows=recording_rows,
        animal_rows=animal_rows,
        group_rows=group_rows,
    )


def run_isolated_recordings(
    args: argparse.Namespace,
    train_cfg: dict[str, Any],
    inference_cfg: dict[str, Any],
    animals: list[str],
    session_dirs: list[Path],
) -> None:
    worker_root = args.out / "_recording_workers"
    worker_root.mkdir(parents=True, exist_ok=True)

    results: list[RecordingResult] = []
    predicted_event_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    global_frame_curves = BinnedCurveAccumulator()
    global_chunk_curves = BinnedCurveAccumulator()

    for idx, session_dir in enumerate(session_dirs, start=1):
        session_key = f"{session_dir.parent.name}/{session_dir.name}"
        safe_key = f"{session_dir.parent.name}_{session_dir.name}".replace("/", "__").replace("\\", "__")
        worker_out = worker_root / safe_key
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--data-dir",
            str(args.data_dir),
            "--train-config",
            str(args.train_config),
            "--inference-config",
            str(args.inference_config),
            "--out",
            str(worker_out),
            "--split",
            args.split,
            "--device",
            args.device,
            "--sessions",
            session_key,
        ]
        if args.save_probabilities:
            command.append("--save-probabilities")

        print(f"[INFO] [{idx}/{len(session_dirs)}] Isolated evaluation: {session_key}")
        completed = None
        for attempt in range(1, int(args.worker_retries) + 2):
            completed = subprocess.run(
                command,
                cwd=PROJECT_ROOT,
                text=True,
                capture_output=True,
            )
            if completed.returncode == 0:
                break
            print(
                f"[WARN] Worker failed for {session_key}, "
                f"attempt {attempt}/{int(args.worker_retries) + 1}"
            )
            if completed.stdout:
                print(completed.stdout)
            if completed.stderr:
                print(completed.stderr)

        if completed is None or completed.returncode != 0:
            raise SystemExit(completed.returncode if completed else 1)
        if completed.stdout:
            print(completed.stdout)

        worker_metrics = json.loads((worker_out / "metrics.json").read_text(encoding="utf-8"))
        worker_recordings = [result_from_dict(item) for item in worker_metrics["recordings"]]
        results.extend(worker_recordings)
        predicted_event_rows.extend(read_csv_rows(worker_out / "predicted_events.csv"))
        match_rows.extend(read_csv_rows(worker_out / "event_matches.csv"))

        frame_bins = BinnedCurveAccumulator.from_jsonable(worker_metrics["curve_bins"]["frame"])
        chunk_bins = BinnedCurveAccumulator.from_jsonable(worker_metrics["curve_bins"]["chunk_1hz"])
        global_frame_curves.pos += frame_bins.pos
        global_frame_curves.neg += frame_bins.neg
        global_chunk_curves.pos += chunk_bins.pos
        global_chunk_curves.neg += chunk_bins.neg

    write_final_outputs(
        args=args,
        train_cfg=train_cfg,
        inference_cfg=inference_cfg,
        animals=animals,
        results=results,
        predicted_event_rows=predicted_event_rows,
        match_rows=match_rows,
        global_frame_curves=global_frame_curves,
        global_chunk_curves=global_chunk_curves,
    )
    print(f"[INFO] Wrote report: {args.out / 'report.md'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Final test metrics/report generator.")
    parser.add_argument("--data-dir", default="data/processed2", type=Path)
    parser.add_argument("--train-config", default="experiments/config_processed2.yaml", type=Path)
    parser.add_argument("--inference-config", default="inference_config.yaml", type=Path)
    parser.add_argument("--out", default="reports/final_test", type=Path)
    parser.add_argument("--split", default="test", choices=["test", "val", "train"])
    parser.add_argument(
        "--sessions",
        default=None,
        help="Comma-separated animal/session keys to evaluate, for example Dex4x5/BL_30Mart.",
    )
    parser.add_argument("--device", default=None, help="Override inference device, e.g. cuda or cpu.")
    parser.add_argument("--max-recordings", type=int, default=None, help="Optional smoke-test limit.")
    parser.add_argument("--save-probabilities", action="store_true")
    parser.add_argument(
        "--isolate-recordings",
        action="store_true",
        help="Evaluate each recording in a separate subprocess and aggregate outputs.",
    )
    parser.add_argument(
        "--worker-retries",
        type=int,
        default=2,
        help="Retries per isolated recording worker.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.data_dir = resolve_path(args.data_dir)
    args.train_config = resolve_path(args.train_config)
    args.inference_config = resolve_path(args.inference_config)
    args.out = resolve_path(args.out)
    args.out.mkdir(parents=True, exist_ok=True)

    train_cfg = load_yaml(args.train_config)
    inference_cfg = load_yaml(args.inference_config)

    split_key = f"{args.split}_animals"
    animals = list(train_cfg["data"].get(split_key, []))
    if not animals:
        raise ValueError(f"No animals configured for split '{args.split}' in {args.train_config}")

    checkpoint = resolve_path(str(inference_cfg["checkpoint"]))
    inference_cfg = dict(inference_cfg)
    inference_cfg["checkpoint"] = str(checkpoint)

    device = args.device or inference_cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    if device == "cuda" and not torch.cuda.is_available():
        print("[WARNING] CUDA requested but unavailable; using CPU.")
        device = "cpu"
    args.device = device

    session_dirs = session_dirs_for_animals(args.data_dir, animals)
    session_filter = None
    if args.sessions:
        session_filter = [item.strip() for item in args.sessions.split(",") if item.strip()]
        session_dirs = filter_session_dirs(session_dirs, session_filter)
    if args.max_recordings is not None:
        session_dirs = session_dirs[: args.max_recordings]
    if not session_dirs:
        raise RuntimeError("No processed recordings found for evaluation.")

    print(f"[INFO] Split: {args.split}; animals: {animals}")
    print(f"[INFO] Recordings: {len(session_dirs)}")
    print(f"[INFO] Model: {inference_cfg['model_name']} from {checkpoint}")
    print(f"[INFO] Device: {device}")

    pharmacology_groups = load_pharmacology_groups(args.data_dir)

    if args.isolate_recordings and len(session_dirs) > 1:
        run_isolated_recordings(
            args=args,
            train_cfg=train_cfg,
            inference_cfg=inference_cfg,
            animals=animals,
            session_dirs=session_dirs,
        )
        return

    model = load_model(str(inference_cfg["model_name"]), str(checkpoint))

    results: list[RecordingResult] = []
    predicted_event_rows: list[dict[str, Any]] = []
    match_rows: list[dict[str, Any]] = []
    global_frame_curves = BinnedCurveAccumulator()
    global_chunk_curves = BinnedCurveAccumulator()

    for idx, session_dir in enumerate(session_dirs, start=1):
        print(f"[INFO] [{idx}/{len(session_dirs)}] Evaluating {session_dir.parent.name}/{session_dir.name}")
        result, pred_rows, rec_match_rows = evaluate_recording(
            session_dir=session_dir,
            model=model,
            inference_cfg=inference_cfg,
            device=device,
            save_probabilities=bool(args.save_probabilities),
            probabilities_dir=args.out / "probabilities",
            global_frame_curves=global_frame_curves,
            global_chunk_curves=global_chunk_curves,
            pharmacology_groups=pharmacology_groups,
        )
        results.append(result)
        predicted_event_rows.extend(pred_rows)
        match_rows.extend(rec_match_rows)
        print(
            "[INFO] "
            f"{result.recording_id}: frame_f1={result.frame_metrics['f1']:.4f}, "
            f"event_f1_standard={result.event_metrics['standard']['f1']:.4f}, "
            f"pred_events={result.pred_event_count}, gt_events={result.gt_event_count}"
        )

    write_final_outputs(
        args=args,
        train_cfg=train_cfg,
        inference_cfg=inference_cfg,
        animals=animals,
        results=results,
        predicted_event_rows=predicted_event_rows,
        match_rows=match_rows,
        global_frame_curves=global_frame_curves,
        global_chunk_curves=global_chunk_curves,
    )
    print(f"[INFO] Wrote report: {args.out / 'report.md'}")


if __name__ == "__main__":
    main()
