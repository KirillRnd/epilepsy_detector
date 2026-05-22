#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-level normalization audit for processed rat EEG/ECoG recordings.

This script intentionally does NOT run a model, thresholds, hysteresis,
frame metrics or event metrics.  It audits only how simple normalization
schemes move recording-level statistics in low-dimensional spaces.

Design principles:
  * Expensive step is done once: compute raw statistics per recording/channel.
  * Normalized statistics are derived analytically from raw stats + center/scale.
  * Report.md is the primary result; figures are a small supporting set.
  * Logical channels are fixed to three slots: FrL, FrR, OcR_Hipp.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import shutil
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import yaml

import matplotlib
matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt


LOGICAL_CHANNELS = ["FrL", "FrR", "OcR_Hipp"]
JOINT_CHANNEL = "__JOINT__"
EPS_DEFAULT = 1e-12


# -----------------------------------------------------------------------------
# Config / paths
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data


def deep_get(d: dict[str, Any], keys: Iterable[str], default: Any = None) -> Any:
    cur: Any = d
    for k in keys:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def resolve_data_dir(args: argparse.Namespace, cfg: dict[str, Any]) -> Path:
    raw = args.data_dir or deep_get(cfg, ["data", "data_dir"], "data/processed2")
    return Path(raw).expanduser().resolve()


def stable_seed(*parts: str) -> int:
    text = "||".join(str(p) for p in parts)
    h = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]
    return int(h, 16)


# -----------------------------------------------------------------------------
# Recording discovery / metadata
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Recording:
    animal_id: str
    session_id: str
    path: Path
    signal_path: Path
    mask_path: Path
    metadata_path: Path | None

    @property
    def recording_id(self) -> str:
        return f"{self.animal_id}/{self.session_id}"


def find_recordings(data_dir: Path) -> list[Recording]:
    records: list[Recording] = []
    if not data_dir.exists():
        raise FileNotFoundError(f"data_dir does not exist: {data_dir}")

    for signal_path in sorted(data_dir.glob("*/*/processed_signals.npy")):
        rec_dir = signal_path.parent
        animal_id = rec_dir.parent.name
        session_id = rec_dir.name
        mask_path = rec_dir / "seizure_mask.npy"
        if not mask_path.exists():
            print(f"[WARN] skip {animal_id}/{session_id}: missing seizure_mask.npy")
            continue
        meta = rec_dir / "conversion_metadata.json"
        records.append(
            Recording(
                animal_id=animal_id,
                session_id=session_id,
                path=rec_dir,
                signal_path=signal_path,
                mask_path=mask_path,
                metadata_path=meta if meta.exists() else None,
            )
        )
    return records


def load_metadata(rec: Recording) -> dict[str, Any]:
    if rec.metadata_path is None:
        return {}
    try:
        with rec.metadata_path.open("r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        print(f"[WARN] failed to read metadata for {rec.recording_id}: {e}")
        return {}


def canonical_channel_name(raw_name: str | None, index: int) -> str | None:
    """Map repository channel names to three logical channel slots."""
    if raw_name is None:
        return LOGICAL_CHANNELS[index] if index < len(LOGICAL_CHANNELS) else None
    key = str(raw_name).strip().lower().replace("-", "").replace("_", "").replace("/", "")
    if key in {"frl", "frontleft"}:
        return "FrL"
    if key in {"frr", "frontright"}:
        return "FrR"
    if key in {"ocr", "hipp", "hip", "ocright", "hippocampus"}:
        return "OcR_Hipp"
    return LOGICAL_CHANNELS[index] if index < len(LOGICAL_CHANNELS) else None


def logical_channel_indices(arr: np.ndarray, metadata: dict[str, Any]) -> dict[str, int]:
    raw_names = metadata.get("channel_names") or []
    mapping: dict[str, int] = {}
    n_channels = int(arr.shape[0])
    for idx in range(n_channels):
        raw_name = raw_names[idx] if idx < len(raw_names) else None
        logical = canonical_channel_name(raw_name, idx)
        if logical is None:
            continue
        # Keep first occurrence; this protects against accidental duplicate names.
        mapping.setdefault(logical, idx)
    return mapping


# -----------------------------------------------------------------------------
# Raw statistics
# -----------------------------------------------------------------------------


def finite_moments(x: np.ndarray) -> tuple[int, float, float, float, float]:
    """Return n, sum, sumsq, min, max over finite values."""
    xf = np.asarray(x, dtype=np.float64)
    finite = np.isfinite(xf)
    if not finite.any():
        return 0, math.nan, math.nan, math.nan, math.nan
    v = xf[finite]
    return int(v.size), float(v.sum()), float(np.square(v).sum()), float(v.min()), float(v.max())


def stats_from_moments(n: int, s: float, ss: float) -> tuple[float, float, float]:
    if n <= 0 or not np.isfinite(s) or not np.isfinite(ss):
        return math.nan, math.nan, math.nan
    mean = s / n
    var = max(0.0, ss / n - mean * mean)
    std = math.sqrt(var)
    rms = math.sqrt(max(0.0, ss / n))
    return float(mean), float(std), float(rms)


def sample_indices_for_segment(
    n: int,
    mask: np.ndarray,
    segment_type: str,
    max_samples: int,
    rng: np.random.Generator,
) -> np.ndarray:
    max_samples = int(max_samples)
    if n <= 0 or max_samples <= 0:
        return np.asarray([], dtype=np.int64)

    if segment_type == "all":
        k = min(n, max_samples)
        return np.sort(rng.choice(n, size=k, replace=False)).astype(np.int64)

    if segment_type == "seizure":
        idx = np.flatnonzero(mask)
        if idx.size <= max_samples:
            return idx.astype(np.int64)
        return np.sort(rng.choice(idx, size=max_samples, replace=False)).astype(np.int64)

    if segment_type == "background":
        n_bg = int(n - np.count_nonzero(mask))
        if n_bg <= 0:
            return np.asarray([], dtype=np.int64)
        k = min(n_bg, max_samples)
        # Rejection sampling avoids materializing flatnonzero(~mask) for long recordings.
        out: list[np.ndarray] = []
        got = 0
        attempts = 0
        while got < k and attempts < 50:
            batch = int(min(n, max((k - got) * 3, 4096)))
            cand = rng.integers(0, n, size=batch, endpoint=False, dtype=np.int64)
            cand = cand[~mask[cand]]
            if cand.size:
                out.append(cand)
                got += cand.size
            attempts += 1
        if not out:
            return np.asarray([], dtype=np.int64)
        idx = np.unique(np.concatenate(out))
        if idx.size > k:
            idx = rng.choice(idx, size=k, replace=False)
        return np.sort(idx).astype(np.int64)

    raise ValueError(f"unknown segment_type: {segment_type}")


def robust_sample_stats(values: np.ndarray, mad_scale: float) -> dict[str, float]:
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return {
            "median": math.nan,
            "mad": math.nan,
            "mad_scaled": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "iqr": math.nan,
        }
    p05, p25, med, p75, p95 = np.percentile(v, [5, 25, 50, 75, 95])
    mad = float(np.median(np.abs(v - med)))
    return {
        "median": float(med),
        "mad": mad,
        "mad_scaled": float(mad * mad_scale),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "iqr": float(p75 - p25),
    }


def segment_moments_from_all_and_seizure(
    x: np.ndarray,
    mask: np.ndarray,
    segment_type: str,
) -> tuple[int, float, float, float, float, float, float]:
    """Exact mean/std/rms components for all/background/seizure.

    Returns n, sum, sumsq, finite_ratio, nan_ratio, inf_ratio, segment_positive_fraction.
    For background, sums are derived from all - seizure to avoid copying background.
    """
    n_total = int(x.shape[0])
    finite_all = np.isfinite(x)
    finite_count_all = int(np.count_nonzero(finite_all))
    nan_count_all = int(np.count_nonzero(np.isnan(x)))
    inf_count_all = int(np.count_nonzero(np.isinf(x)))

    if segment_type == "all":
        n, s, ss, _mn, _mx = finite_moments(x)
        denom = max(1, n_total)
        return n, s, ss, finite_count_all / denom, nan_count_all / denom, inf_count_all / denom, float(np.mean(mask))

    seiz_x = x[mask]
    n_seiz_total = int(seiz_x.shape[0])
    n_seiz, s_seiz, ss_seiz, _mn, _mx = finite_moments(seiz_x)

    if segment_type == "seizure":
        denom = max(1, n_seiz_total)
        return (
            n_seiz,
            s_seiz,
            ss_seiz,
            float(np.isfinite(seiz_x).sum() / denom),
            float(np.isnan(seiz_x).sum() / denom),
            float(np.isinf(seiz_x).sum() / denom),
            1.0 if n_seiz_total else 0.0,
        )

    if segment_type == "background":
        n_all, s_all, ss_all, _mn, _mx = finite_moments(x)
        n_bg = n_all - n_seiz
        s_bg = s_all - s_seiz
        ss_bg = ss_all - ss_seiz
        n_bg_total = n_total - n_seiz_total
        # Ratios for background are not critical; derive finite ratio exactly enough.
        denom = max(1, n_bg_total)
        finite_bg = max(0, finite_count_all - int(np.isfinite(seiz_x).sum()))
        nan_bg = max(0, nan_count_all - int(np.isnan(seiz_x).sum()))
        inf_bg = max(0, inf_count_all - int(np.isinf(seiz_x).sum()))
        return n_bg, s_bg, ss_bg, finite_bg / denom, nan_bg / denom, inf_bg / denom, 0.0

    raise ValueError(f"unknown segment_type: {segment_type}")


def compute_raw_stats_for_record(
    rec: Recording,
    cfg: dict[str, Any],
    segments_needed: list[str],
) -> list[dict[str, Any]]:
    arr = np.load(rec.signal_path, mmap_mode="r")
    mask = np.load(rec.mask_path, mmap_mode="r").astype(bool)
    if arr.ndim != 2:
        raise ValueError(f"{rec.recording_id}: expected processed_signals.npy shape (C, T), got {arr.shape}")
    n_samples = min(int(arr.shape[1]), int(mask.shape[0]))
    if n_samples <= 0:
        return []
    if arr.shape[1] != n_samples:
        arr = arr[:, :n_samples]
    if mask.shape[0] != n_samples:
        mask = mask[:n_samples]

    metadata = load_metadata(rec)
    channel_map = logical_channel_indices(arr, metadata)
    sfreq = metadata.get("sampling_freq", metadata.get("sfreq", math.nan))
    duration_s = float(metadata.get("duration", n_samples / sfreq if sfreq and np.isfinite(sfreq) else math.nan))

    max_samples = int(deep_get(cfg, ["stats", "robust_sample_max"], 200_000))
    mad_scale = float(deep_get(cfg, ["normalization", "mad_scale"], 1.4826))

    rows: list[dict[str, Any]] = []
    per_segment_channel_samples: dict[str, list[np.ndarray]] = {s: [] for s in segments_needed}
    per_segment_joint_moments: dict[str, list[tuple[int, float, float]]] = {s: [] for s in segments_needed}

    for logical_ch in LOGICAL_CHANNELS:
        if logical_ch not in channel_map:
            continue
        ch_idx = channel_map[logical_ch]
        x = np.asarray(arr[ch_idx, :n_samples], dtype=np.float64)

        for segment_type in segments_needed:
            rng = np.random.default_rng(stable_seed(rec.recording_id, logical_ch, segment_type))
            n, s, ss, finite_ratio, nan_ratio, inf_ratio, segment_positive_fraction = segment_moments_from_all_and_seizure(
                x, mask, segment_type
            )
            mean, std, rms = stats_from_moments(n, s, ss)
            sample_idx = sample_indices_for_segment(n_samples, mask, segment_type, max_samples, rng)
            sample_values = x[sample_idx] if sample_idx.size else np.asarray([], dtype=np.float64)
            robust = robust_sample_stats(sample_values, mad_scale=mad_scale)
            per_segment_channel_samples[segment_type].append(sample_values)
            per_segment_joint_moments[segment_type].append((n, s, ss))

            rows.append(
                {
                    "recording_id": rec.recording_id,
                    "animal_id": rec.animal_id,
                    "session_id": rec.session_id,
                    "channel": logical_ch,
                    "channel_index": int(ch_idx),
                    "segment_type": segment_type,
                    "is_joint": False,
                    "n_samples": int(n),
                    "duration_s": duration_s,
                    "sampling_freq": sfreq,
                    "seizure_fraction": float(np.mean(mask)),
                    "mean": mean,
                    "std": std,
                    "rms": rms,
                    "median": robust["median"],
                    "mad": robust["mad"],
                    "mad_scaled": robust["mad_scaled"],
                    "p05": robust["p05"],
                    "p25": robust["p25"],
                    "p75": robust["p75"],
                    "p95": robust["p95"],
                    "iqr": robust["iqr"],
                    "finite_ratio": finite_ratio,
                    "nan_ratio": nan_ratio,
                    "inf_ratio": inf_ratio,
                    "sampled_robust_stats": bool(n > max_samples),
                }
            )

    # Joint channel rows are used for joint normalization parameters.
    for segment_type in segments_needed:
        moments = per_segment_joint_moments[segment_type]
        if not moments:
            continue
        n_joint = int(sum(m[0] for m in moments))
        s_joint = float(sum(m[1] for m in moments))
        ss_joint = float(sum(m[2] for m in moments))
        mean, std, rms = stats_from_moments(n_joint, s_joint, ss_joint)
        samples = [v for v in per_segment_channel_samples[segment_type] if v.size]
        joint_sample = np.concatenate(samples) if samples else np.asarray([], dtype=np.float64)
        if joint_sample.size > max_samples:
            rng = np.random.default_rng(stable_seed(rec.recording_id, JOINT_CHANNEL, segment_type))
            joint_sample = rng.choice(joint_sample, size=max_samples, replace=False)
        robust = robust_sample_stats(joint_sample, mad_scale=mad_scale)
        rows.append(
            {
                "recording_id": rec.recording_id,
                "animal_id": rec.animal_id,
                "session_id": rec.session_id,
                "channel": JOINT_CHANNEL,
                "channel_index": -1,
                "segment_type": segment_type,
                "is_joint": True,
                "n_samples": n_joint,
                "duration_s": duration_s,
                "sampling_freq": sfreq,
                "seizure_fraction": float(np.mean(mask)),
                "mean": mean,
                "std": std,
                "rms": rms,
                "median": robust["median"],
                "mad": robust["mad"],
                "mad_scaled": robust["mad_scaled"],
                "p05": robust["p05"],
                "p25": robust["p25"],
                "p75": robust["p75"],
                "p95": robust["p95"],
                "iqr": robust["iqr"],
                "finite_ratio": math.nan,
                "nan_ratio": math.nan,
                "inf_ratio": math.nan,
                "sampled_robust_stats": bool(n_joint > max_samples),
            }
        )

    return rows


def compute_raw_stats(data_dir: Path, out_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    report_segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    fit_segments = {str(n.get("fit_segment", "all")) for n in deep_get(cfg, ["normalization", "methods"], [])}
    segments_needed = sorted(set(report_segments) | fit_segments | {"all"})

    records = find_recordings(data_dir)
    print(f"Found {len(records)} recordings in {data_dir}")
    all_rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec.recording_id}")
        try:
            all_rows.extend(compute_raw_stats_for_record(rec, cfg, segments_needed=segments_needed))
        except Exception as e:
            print(f"[WARN] failed {rec.recording_id}: {e}")

    raw_df = pd.DataFrame(all_rows)
    ensure_dir(out_dir)
    raw_path = out_dir / "raw_stats.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"[OK] wrote {raw_path}")
    return raw_df


# -----------------------------------------------------------------------------
# Normalization effects from raw stats
# -----------------------------------------------------------------------------


def safe_scale(value: float, eps: float) -> float:
    if value is None or not np.isfinite(value) or abs(value) < eps:
        return math.nan
    return float(value)


def fallback_robust_scale(row: pd.Series, eps: float) -> float:
    for key, factor in [("mad_scaled", 1.0), ("iqr", 1.0 / 1.349), ("std", 1.0)]:
        val = row.get(key, math.nan)
        scale = val * factor
        if np.isfinite(scale) and abs(scale) >= eps:
            return float(scale)
    return math.nan


def build_param_lookup(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> dict[tuple[str, str, str, str], tuple[float, float, str]]:
    """Return (normalization_id, recording_id, channel, fit_segment) -> center, scale, fit_scope."""
    eps = float(deep_get(cfg, ["normalization", "eps"], EPS_DEFAULT))
    lookup: dict[tuple[str, str, str, str], tuple[float, float, str]] = {}
    indexed = raw_df.set_index(["recording_id", "channel", "segment_type"], drop=False)

    for method in deep_get(cfg, ["normalization", "methods"], []):
        name = str(method["name"])
        kind = str(method.get("kind", "none"))
        fit_segment = str(method.get("fit_segment", "all"))
        fit_scope = str(method.get("fit_scope", kind))

        if kind == "none":
            for rec_id in raw_df["recording_id"].drop_duplicates():
                for ch in LOGICAL_CHANNELS:
                    lookup[(name, rec_id, ch, fit_segment)] = (0.0, 1.0, fit_scope)
            continue

        for rec_id in raw_df["recording_id"].drop_duplicates():
            if "joint" in kind:
                key = (rec_id, JOINT_CHANNEL, fit_segment)
                if key not in indexed.index:
                    continue
                row = indexed.loc[key]
                if isinstance(row, pd.DataFrame):
                    row = row.iloc[0]
                if "robust" in kind:
                    center = float(row.get("median", math.nan))
                    scale = fallback_robust_scale(row, eps)
                else:
                    center = float(row.get("mean", math.nan))
                    scale = safe_scale(float(row.get("std", math.nan)), eps)
                for ch in LOGICAL_CHANNELS:
                    lookup[(name, rec_id, ch, fit_segment)] = (center, scale, fit_scope)
            else:
                for ch in LOGICAL_CHANNELS:
                    key = (rec_id, ch, fit_segment)
                    if key not in indexed.index:
                        continue
                    row = indexed.loc[key]
                    if isinstance(row, pd.DataFrame):
                        row = row.iloc[0]
                    if "robust" in kind:
                        center = float(row.get("median", math.nan))
                        scale = fallback_robust_scale(row, eps)
                    else:
                        center = float(row.get("mean", math.nan))
                        scale = safe_scale(float(row.get("std", math.nan)), eps)
                    lookup[(name, rec_id, ch, fit_segment)] = (center, scale, fit_scope)
    return lookup


def derive_effects(raw_df: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    report_segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    raw_channels = raw_df[(raw_df["channel"].isin(LOGICAL_CHANNELS)) & (raw_df["segment_type"].isin(report_segments))]
    param_lookup = build_param_lookup(raw_df, cfg)

    rows: list[dict[str, Any]] = []
    for _, r in raw_channels.iterrows():
        rec_id = str(r["recording_id"])
        ch = str(r["channel"])
        segment_type = str(r["segment_type"])
        for method in deep_get(cfg, ["normalization", "methods"], []):
            name = str(method["name"])
            fit_segment = str(method.get("fit_segment", "all"))
            params = param_lookup.get((name, rec_id, ch, fit_segment))
            if params is None:
                continue
            center, scale, fit_scope = params
            if not np.isfinite(center) or not np.isfinite(scale) or scale == 0:
                continue
            raw_mean = float(r["mean"])
            raw_std = float(r["std"])
            raw_rms = float(r["rms"])
            raw_median = float(r["median"])
            raw_iqr = float(r["iqr"])
            norm_mean = (raw_mean - center) / scale
            norm_std = raw_std / scale
            second_moment_after_center = max(0.0, raw_rms * raw_rms - 2.0 * center * raw_mean + center * center)
            norm_rms = math.sqrt(second_moment_after_center) / scale
            norm_median = (raw_median - center) / scale if np.isfinite(raw_median) else math.nan
            norm_iqr = raw_iqr / scale if np.isfinite(raw_iqr) else math.nan

            rows.append(
                {
                    "recording_id": rec_id,
                    "animal_id": r["animal_id"],
                    "session_id": r["session_id"],
                    "channel": ch,
                    "segment_type": segment_type,
                    "normalization_id": name,
                    "fit_scope": fit_scope,
                    "fit_segment": fit_segment,
                    "center": center,
                    "scale": scale,
                    "raw_mean": raw_mean,
                    "raw_std": raw_std,
                    "raw_rms": raw_rms,
                    "raw_median": raw_median,
                    "raw_iqr": raw_iqr,
                    "norm_mean": norm_mean,
                    "norm_std": norm_std,
                    "norm_rms": norm_rms,
                    "norm_median": norm_median,
                    "norm_iqr": norm_iqr,
                    "delta_mean": norm_mean - raw_mean,
                    "delta_std": norm_std - raw_std,
                    "n_samples": int(r["n_samples"]),
                    "seizure_fraction": float(r["seizure_fraction"]),
                }
            )

    effects_df = pd.DataFrame(rows)
    path = out_dir / "normalization_effects.csv"
    effects_df.to_csv(path, index=False)
    print(f"[OK] wrote {path}")
    return effects_df


# -----------------------------------------------------------------------------
# Summary tables
# -----------------------------------------------------------------------------


def finite_std(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return float(np.std(v)) if v.size else math.nan


def finite_mean(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return float(np.mean(v)) if v.size else math.nan


def finite_median(values: pd.Series) -> float:
    v = pd.to_numeric(values, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)
    return float(np.median(v)) if v.size else math.nan


def ratio(a: float, b: float) -> float:
    return float(a / b) if np.isfinite(a) and np.isfinite(b) and abs(b) > 0 else math.nan


def summarize_effects(effects_df: pd.DataFrame, out_dir: Path) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for (norm, seg, ch), g in effects_df.groupby(["normalization_id", "segment_type", "channel"], sort=True):
        raw_std_disp = finite_std(g["raw_std"])
        norm_std_disp = finite_std(g["norm_std"])
        raw_iqr_disp = finite_std(g["raw_iqr"])
        norm_iqr_disp = finite_std(g["norm_iqr"])
        raw_rms_disp = finite_std(g["raw_rms"])
        norm_rms_disp = finite_std(g["norm_rms"])
        rows.append(
            {
                "normalization_id": norm,
                "segment_type": seg,
                "channel": ch,
                "n_recordings": int(g["recording_id"].nunique()),
                "raw_centroid_mean": finite_mean(g["raw_mean"]),
                "raw_centroid_std": finite_mean(g["raw_std"]),
                "norm_centroid_mean": finite_mean(g["norm_mean"]),
                "norm_centroid_std": finite_mean(g["norm_std"]),
                "centroid_shift_mean": finite_mean(g["norm_mean"]) - finite_mean(g["raw_mean"]),
                "centroid_shift_std": finite_mean(g["norm_std"]) - finite_mean(g["raw_std"]),
                "raw_std_dispersion": raw_std_disp,
                "norm_std_dispersion": norm_std_disp,
                "std_dispersion_ratio": ratio(norm_std_disp, raw_std_disp),
                "raw_iqr_dispersion": raw_iqr_disp,
                "norm_iqr_dispersion": norm_iqr_disp,
                "iqr_dispersion_ratio": ratio(norm_iqr_disp, raw_iqr_disp),
                "raw_rms_dispersion": raw_rms_disp,
                "norm_rms_dispersion": norm_rms_disp,
                "rms_dispersion_ratio": ratio(norm_rms_disp, raw_rms_disp),
                "raw_abs_mean_median": finite_median(g["raw_mean"].abs()),
                "norm_abs_mean_median": finite_median(g["norm_mean"].abs()),
                "raw_std_median": finite_median(g["raw_std"]),
                "norm_std_median": finite_median(g["norm_std"]),
            }
        )
    df = pd.DataFrame(rows)
    path = out_dir / "normalization_summary.csv"
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path}")
    return df


def coeff_var(vals: np.ndarray) -> float:
    vals = np.asarray(vals, dtype=float)
    vals = vals[np.isfinite(vals)]
    if vals.size == 0:
        return math.nan
    m = float(np.mean(vals))
    if abs(m) < EPS_DEFAULT:
        return math.nan
    return float(np.std(vals) / abs(m))


def channel_balance(effects_df: pd.DataFrame, out_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for (norm, seg, rec_id), g in effects_df.groupby(["normalization_id", "segment_type", "recording_id"], sort=True):
        pivot = g.set_index("channel")
        if not all(ch in pivot.index for ch in LOGICAL_CHANNELS):
            continue
        raw_stds = np.asarray([float(pivot.loc[ch, "raw_std"]) for ch in LOGICAL_CHANNELS], dtype=float)
        norm_stds = np.asarray([float(pivot.loc[ch, "norm_std"]) for ch in LOGICAL_CHANNELS], dtype=float)
        def lr(a: float, b: float) -> float:
            return float(np.log(a / b)) if a > 0 and b > 0 else math.nan
        animal = str(g["animal_id"].iloc[0])
        session = str(g["session_id"].iloc[0])
        rows.append(
            {
                "normalization_id": norm,
                "segment_type": seg,
                "recording_id": rec_id,
                "animal_id": animal,
                "session_id": session,
                "raw_channel_std_cv": coeff_var(raw_stds),
                "norm_channel_std_cv": coeff_var(norm_stds),
                "raw_channel_std_range": float(np.max(raw_stds) - np.min(raw_stds)),
                "norm_channel_std_range": float(np.max(norm_stds) - np.min(norm_stds)),
                "raw_log_std_FrL_FrR": lr(raw_stds[0], raw_stds[1]),
                "norm_log_std_FrL_FrR": lr(norm_stds[0], norm_stds[1]),
                "raw_log_std_FrL_OcR_Hipp": lr(raw_stds[0], raw_stds[2]),
                "norm_log_std_FrL_OcR_Hipp": lr(norm_stds[0], norm_stds[2]),
                "raw_log_std_FrR_OcR_Hipp": lr(raw_stds[1], raw_stds[2]),
                "norm_log_std_FrR_OcR_Hipp": lr(norm_stds[1], norm_stds[2]),
            }
        )
    bal_df = pd.DataFrame(rows)
    bal_path = out_dir / "channel_balance.csv"
    bal_df.to_csv(bal_path, index=False)
    print(f"[OK] wrote {bal_path}")

    srows: list[dict[str, Any]] = []
    for (norm, seg), g in bal_df.groupby(["normalization_id", "segment_type"], sort=True):
        raw_cv = finite_median(g["raw_channel_std_cv"])
        norm_cv = finite_median(g["norm_channel_std_cv"])
        raw_log_disp = finite_mean(pd.Series([
            finite_std(g["raw_log_std_FrL_FrR"]),
            finite_std(g["raw_log_std_FrL_OcR_Hipp"]),
            finite_std(g["raw_log_std_FrR_OcR_Hipp"]),
        ]))
        norm_log_disp = finite_mean(pd.Series([
            finite_std(g["norm_log_std_FrL_FrR"]),
            finite_std(g["norm_log_std_FrL_OcR_Hipp"]),
            finite_std(g["norm_log_std_FrR_OcR_Hipp"]),
        ]))
        srows.append(
            {
                "normalization_id": norm,
                "segment_type": seg,
                "n_recordings": int(g["recording_id"].nunique()),
                "raw_channel_std_cv_median": raw_cv,
                "norm_channel_std_cv_median": norm_cv,
                "channel_std_cv_ratio": ratio(norm_cv, raw_cv),
                "raw_log_std_ratio_dispersion": raw_log_disp,
                "norm_log_std_ratio_dispersion": norm_log_disp,
                "log_std_ratio_dispersion_ratio": ratio(norm_log_disp, raw_log_disp),
            }
        )
    bal_summary = pd.DataFrame(srows)
    summary_path = out_dir / "channel_balance_summary.csv"
    bal_summary.to_csv(summary_path, index=False)
    print(f"[OK] wrote {summary_path}")
    return bal_df, bal_summary


def overall_summary(norm_summary: pd.DataFrame, bal_summary: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    methods = [m["name"] for m in deep_get(cfg, ["normalization", "methods"], []) if m.get("kind") != "none"]
    for norm in methods:
        for seg in deep_get(cfg, ["segments", "report"], ["background", "seizure"]):
            g = norm_summary[(norm_summary["normalization_id"] == norm) & (norm_summary["segment_type"] == seg)]
            b = bal_summary[(bal_summary["normalization_id"] == norm) & (bal_summary["segment_type"] == seg)]
            if g.empty:
                continue
            channel_cv_ratio = finite_median(b["channel_std_cv_ratio"]) if not b.empty else math.nan
            rows.append(
                {
                    "normalization_id": norm,
                    "segment_type": seg,
                    "std_dispersion_ratio_median": finite_median(g["std_dispersion_ratio"]),
                    "iqr_dispersion_ratio_median": finite_median(g["iqr_dispersion_ratio"]),
                    "rms_dispersion_ratio_median": finite_median(g["rms_dispersion_ratio"]),
                    "norm_std_median_median": finite_median(g["norm_std_median"]),
                    "norm_abs_mean_median": finite_median(g["norm_abs_mean_median"]),
                    "channel_std_cv_ratio": channel_cv_ratio,
                    "norm_channel_std_cv_median": finite_median(b["norm_channel_std_cv_median"]) if not b.empty else math.nan,
                }
            )
    df = pd.DataFrame(rows)
    path = out_dir / "normalization_overall_summary.csv"
    df.to_csv(path, index=False)
    print(f"[OK] wrote {path}")
    return df


# -----------------------------------------------------------------------------
# Plotting: small selected figure set
# -----------------------------------------------------------------------------


def animal_colors(values: Iterable[str]) -> dict[str, Any]:
    vals = sorted(set(str(v) for v in values))
    cmap = plt.get_cmap("tab20")
    return {v: cmap(i % cmap.N) for i, v in enumerate(vals)}


def safe_save(fig: plt.Figure, path: Path, dpi: int) -> None:
    ensure_dir(path.parent)
    try:
        fig.savefig(path, dpi=dpi)
    finally:
        plt.close(fig)


def scatter_by_animal(ax: plt.Axes, df: pd.DataFrame, x: str, y: str, colors: dict[str, Any], marker: str = "o", size: float = 16.0) -> None:
    for animal, g in df.groupby("animal_id", sort=True):
        ax.scatter(g[x], g[y], s=size, alpha=0.75, marker=marker, color=colors.get(str(animal)), linewidths=0.0)


def plot_raw_geometry(effects_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    base = effects_df[effects_df["normalization_id"] == "none"].copy()
    if base.empty:
        # Fall back to first normalization for raw fields; raw_* are same for all methods.
        base = effects_df.drop_duplicates(["recording_id", "channel", "segment_type"])
    colors = animal_colors(base["animal_id"])
    for seg in deep_get(cfg, ["segments", "report"], ["background", "seizure"]):
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.4), squeeze=False)
        for j, ch in enumerate(LOGICAL_CHANNELS):
            ax = axes[0, j]
            g = base[(base["segment_type"] == seg) & (base["channel"] == ch)]
            scatter_by_animal(ax, g, "raw_mean", "raw_std", colors, marker="o")
            ax.set_title(ch)
            ax.set_xlabel("raw mean")
            if j == 0:
                ax.set_ylabel("raw std")
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"Raw mean/std geometry | {seg}")
        fig.subplots_adjust(left=0.055, right=0.985, bottom=0.18, top=0.78, wspace=0.28)
        path = fig_dir / f"01_raw_geometry__{seg}.png"
        safe_save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_mean_std_movement(effects_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    selected = list(deep_get(cfg, ["plots", "selected_normalizations"], []))
    if not selected:
        selected = [n for n in effects_df["normalization_id"].drop_duplicates().tolist() if n != "none"]
    colors = animal_colors(effects_df["animal_id"])
    for seg in deep_get(cfg, ["segments", "report"], ["background", "seizure"]):
        nrows = len(selected)
        fig, axes = plt.subplots(nrows, 3, figsize=(13, max(2.7 * nrows, 3.0)), squeeze=False)
        for i, norm in enumerate(selected):
            for j, ch in enumerate(LOGICAL_CHANNELS):
                ax = axes[i, j]
                g = effects_df[(effects_df["normalization_id"] == norm) & (effects_df["segment_type"] == seg) & (effects_df["channel"] == ch)]
                for _, r in g.iterrows():
                    c = colors.get(str(r["animal_id"]))
                    ax.plot([r["raw_mean"], r["norm_mean"]], [r["raw_std"], r["norm_std"]], color=c, alpha=0.35, linewidth=0.7)
                    ax.scatter([r["raw_mean"]], [r["raw_std"]], color=c, marker="o", s=10, alpha=0.6, linewidths=0)
                    ax.scatter([r["norm_mean"]], [r["norm_std"]], color=c, marker="x", s=14, alpha=0.8, linewidths=0.7)
                if i == 0:
                    ax.set_title(ch)
                if j == 0:
                    ax.set_ylabel(norm.replace("per_record_", ""))
                if i == nrows - 1:
                    ax.set_xlabel("mean (symlog)")
                if j == 2:
                    ax.text(1.02, 0.5, "std (log)", transform=ax.transAxes, rotation=90, va="center", fontsize=8)
                ax.set_xscale("symlog", linthresh=1e-6)
                ax.set_yscale("log")
                ax.grid(True, alpha=0.25)
        fig.suptitle(f"Mean/std movement: raw=o, normalized=x | {seg}")
        fig.subplots_adjust(left=0.105, right=0.965, bottom=0.075, top=0.91, wspace=0.28, hspace=0.45)
        path = fig_dir / f"02_mean_std_movement__{seg}.png"
        safe_save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_raw_to_norm_std(effects_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    selected = list(deep_get(cfg, ["plots", "selected_normalizations"], []))
    colors = animal_colors(effects_df["animal_id"])
    for seg in deep_get(cfg, ["segments", "report"], ["background", "seizure"]):
        nrows = len(selected)
        fig, axes = plt.subplots(nrows, 3, figsize=(13, max(2.5 * nrows, 3.0)), squeeze=False)
        for i, norm in enumerate(selected):
            for j, ch in enumerate(LOGICAL_CHANNELS):
                ax = axes[i, j]
                g = effects_df[(effects_df["normalization_id"] == norm) & (effects_df["segment_type"] == seg) & (effects_df["channel"] == ch)]
                scatter_by_animal(ax, g, "raw_std", "norm_std", colors, marker="o", size=13)
                if i == 0:
                    ax.set_title(ch)
                if j == 0:
                    ax.set_ylabel(norm.replace("per_record_", ""))
                if i == nrows - 1:
                    ax.set_xlabel("raw std")
                ax.set_xscale("log")
                ax.set_yscale("log")
                ax.grid(True, alpha=0.25)
        fig.suptitle(f"Raw std -> normalized std | {seg}")
        fig.subplots_adjust(left=0.105, right=0.985, bottom=0.075, top=0.91, wspace=0.28, hspace=0.45)
        path = fig_dir / f"03_raw_to_norm_std__{seg}.png"
        safe_save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_channel_balance(bal_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    selected = list(deep_get(cfg, ["plots", "selected_normalizations"], []))
    colors = animal_colors(bal_df["animal_id"] if not bal_df.empty else [])
    for seg in deep_get(cfg, ["segments", "report"], ["background", "seizure"]):
        ncols = len(selected)
        fig, axes = plt.subplots(1, ncols, figsize=(3.3 * ncols, 3.2), squeeze=False)
        for j, norm in enumerate(selected):
            ax = axes[0, j]
            g = bal_df[(bal_df["normalization_id"] == norm) & (bal_df["segment_type"] == seg)]
            scatter_by_animal(ax, g, "raw_channel_std_cv", "norm_channel_std_cv", colors, marker="o", size=15)
            if not g.empty:
                lo = float(np.nanmin([g["raw_channel_std_cv"].min(), g["norm_channel_std_cv"].min()]))
                hi = float(np.nanmax([g["raw_channel_std_cv"].max(), g["norm_channel_std_cv"].max()]))
                if np.isfinite(lo) and np.isfinite(hi) and hi > lo:
                    ax.plot([lo, hi], [lo, hi], linestyle="--", linewidth=0.8, color="0.5", alpha=0.7)
            ax.set_title(norm.replace("per_record_", ""), fontsize=9)
            ax.set_xlabel("raw channel std CV")
            if j == 0:
                ax.set_ylabel("normalized channel std CV")
            ax.grid(True, alpha=0.25)
        fig.suptitle(f"Inter-channel scale balance | {seg}")
        fig.subplots_adjust(left=0.07, right=0.985, bottom=0.18, top=0.78, wspace=0.32)
        path = fig_dir / f"04_channel_balance__{seg}.png"
        safe_save(fig, path, dpi)
        paths.append(path)
    return paths


def plot_dispersion_summary(overall: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    if overall.empty:
        return []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    fig, ax = plt.subplots(1, 1, figsize=(9.5, 4.2))
    methods = list(deep_get(cfg, ["plots", "selected_normalizations"], []))
    x = np.arange(len(methods))
    width = 0.35
    for offset, seg in [(-width / 2, "background"), (width / 2, "seizure")]:
        vals = []
        for m in methods:
            row = overall[(overall["normalization_id"] == m) & (overall["segment_type"] == seg)]
            vals.append(float(row["std_dispersion_ratio_median"].iloc[0]) if not row.empty else math.nan)
        ax.bar(x + offset, vals, width=width, label=seg, alpha=0.8)
    ax.axhline(1.0, color="0.4", linewidth=0.8, linestyle="--")
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace("per_record_", "") for m in methods], rotation=20, ha="right")
    ax.set_ylabel("median std-dispersion ratio after/before")
    ax.set_title("Cluster compression by normalization (lower = stronger compression)")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.09, right=0.98, bottom=0.28, top=0.86)
    path = fig_dir / "05_dispersion_summary.png"
    safe_save(fig, path, dpi)
    return [path]


def make_figures(effects_df: pd.DataFrame, bal_df: pd.DataFrame, overall: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    if not bool(deep_get(cfg, ["plots", "enabled"], True)):
        return []
    fig_dir = ensure_dir(out_dir / "figures" / "main")
    if bool(deep_get(cfg, ["plots", "clean_main_dir"], True)) and fig_dir.exists():
        shutil.rmtree(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_raw_geometry(effects_df, fig_dir, cfg))
    paths.extend(plot_mean_std_movement(effects_df, fig_dir, cfg))
    paths.extend(plot_raw_to_norm_std(effects_df, fig_dir, cfg))
    paths.extend(plot_channel_balance(bal_df, fig_dir, cfg))
    paths.extend(plot_dispersion_summary(overall, fig_dir, cfg))
    return paths


# -----------------------------------------------------------------------------
# Markdown report
# -----------------------------------------------------------------------------


def fmt(x: Any, digits: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(v):
        return "—"
    if v == 0:
        return "0"
    if abs(v) < 1e-3 or abs(v) >= 1e4:
        return f"{v:.{digits}e}"
    return f"{v:.{digits}f}"


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 20, digits: int = 4) -> str:
    if df.empty:
        return "_No rows._\n"
    d = df[columns].head(max_rows).copy()
    lines = []
    lines.append("| " + " | ".join(columns) + " |")
    lines.append("| " + " | ".join(["---"] * len(columns)) + " |")
    for _, row in d.iterrows():
        vals = [fmt(row[c], digits=digits) for c in columns]
        lines.append("| " + " | ".join(vals) + " |")
    if len(df) > max_rows:
        lines.append(f"\n_Showing {max_rows} of {len(df)} rows._")
    return "\n".join(lines) + "\n"


def interpret_method(row: pd.Series) -> str:
    std_r = row.get("std_dispersion_ratio_median", math.nan)
    ch_r = row.get("channel_std_cv_ratio", math.nan)
    bits = []
    if np.isfinite(std_r):
        if std_r < 0.25:
            bits.append("strongly compresses between-recording amplitude spread")
        elif std_r < 0.6:
            bits.append("moderately compresses between-recording amplitude spread")
        elif std_r < 1.1:
            bits.append("mostly preserves between-recording spread")
        else:
            bits.append("increases between-recording spread")
    if np.isfinite(ch_r):
        if ch_r < 0.25:
            bits.append("strongly collapses inter-channel scale differences")
        elif ch_r < 0.7:
            bits.append("reduces inter-channel scale differences")
        elif ch_r < 1.3:
            bits.append("roughly preserves inter-channel scale balance")
        else:
            bits.append("increases inter-channel imbalance")
    return "; ".join(bits) if bits else "not enough data"


def report_markdown(
    raw_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    norm_summary: pd.DataFrame,
    bal_df: pd.DataFrame,
    bal_summary: pd.DataFrame,
    overall: pd.DataFrame,
    figure_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
) -> str:
    report_segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    n_recordings = int(raw_df["recording_id"].nunique())

    # Completeness is based on raw non-joint rows for report segments.
    raw_non_joint = raw_df[(raw_df["channel"].isin(LOGICAL_CHANNELS)) & (raw_df["segment_type"].isin(report_segments))]
    completeness_rows = []
    for seg in report_segments:
        g = raw_non_joint[raw_non_joint["segment_type"] == seg]
        counts = g.groupby("recording_id")["channel"].nunique()
        completeness_rows.append({
            "segment_type": seg,
            "complete_triplets": int((counts == 3).sum()),
            "expected_recordings": n_recordings,
            "incomplete_recordings": int((counts < 3).sum()),
        })
    completeness = pd.DataFrame(completeness_rows)

    # Mean-axis usefulness check.
    base = effects_df[effects_df["normalization_id"] == "none"]
    if base.empty:
        base = effects_df.drop_duplicates(["recording_id", "channel", "segment_type"])
    mean_ratio = finite_median(base["raw_mean"].abs()) / max(finite_median(base["raw_std"]), EPS_DEFAULT)

    ordered_overall = overall.copy()
    if not ordered_overall.empty:
        ordered_overall["interpretation"] = ordered_overall.apply(interpret_method, axis=1)
        ordered_overall = ordered_overall.sort_values(["segment_type", "std_dispersion_ratio_median", "channel_std_cv_ratio"], na_position="last")

    # Worst residual outliers by normalized std distance from channel/segment/norm median.
    outlier_rows = []
    for (norm, seg, ch), g in effects_df[effects_df["normalization_id"] != "none"].groupby(["normalization_id", "segment_type", "channel"]):
        med = finite_median(g["norm_std"])
        if not np.isfinite(med):
            continue
        tmp = g.copy()
        tmp["norm_std_abs_deviation"] = (tmp["norm_std"] - med).abs()
        top = tmp.sort_values("norm_std_abs_deviation", ascending=False).head(2)
        for _, r in top.iterrows():
            outlier_rows.append({
                "normalization_id": norm,
                "segment_type": seg,
                "channel": ch,
                "recording_id": r["recording_id"],
                "norm_std": r["norm_std"],
                "deviation": r["norm_std_abs_deviation"],
            })
    outliers = pd.DataFrame(outlier_rows).sort_values("deviation", ascending=False).head(12) if outlier_rows else pd.DataFrame()

    rel_figs = [p.relative_to(out_dir).as_posix() for p in figure_paths]

    lines: list[str] = []
    lines.append("# Normalization Audit\n")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n")
    lines.append("## Scope\n")
    lines.append("This report audits only signal normalization. It does not run model inference, hysteresis, frame metrics, event metrics, or biological downstream metrics.\n")

    lines.append("## Executive summary\n")
    if mean_ratio < 1e-3:
        lines.append(f"- Raw means are already tiny relative to raw standard deviations: median |mean|/std ≈ `{fmt(mean_ratio)}`. The mean axis is kept for the agreed two-axis geometry, but most visible domain movement is expected along std/RMS/IQR rather than mean.\n")
    else:
        lines.append(f"- Raw means are not negligible relative to raw std: median |mean|/std ≈ `{fmt(mean_ratio)}`. Mean shifts should be inspected directly.\n")
    if not ordered_overall.empty:
        bg = ordered_overall[ordered_overall["segment_type"] == "background"].head(1)
        sz = ordered_overall[ordered_overall["segment_type"] == "seizure"].head(1)
        if not bg.empty:
            r = bg.iloc[0]
            lines.append(f"- Strongest background std-cluster compression in this report: `{r['normalization_id']}` with median std-dispersion ratio `{fmt(r['std_dispersion_ratio_median'])}` and channel-CV ratio `{fmt(r['channel_std_cv_ratio'])}`.\n")
        if not sz.empty:
            r = sz.iloc[0]
            lines.append(f"- Strongest seizure std-cluster compression in this report: `{r['normalization_id']}` with median std-dispersion ratio `{fmt(r['std_dispersion_ratio_median'])}` and channel-CV ratio `{fmt(r['channel_std_cv_ratio'])}`.\n")
    lines.append("- Interpret low dispersion ratios together with channel-balance ratios: a method that compresses recording clusters may still be too aggressive if it collapses inter-channel scale structure.\n")

    lines.append("## Data sanity\n")
    lines.append(f"- Recordings: `{n_recordings}`\n")
    lines.append(f"- Logical channels: `{', '.join(LOGICAL_CHANNELS)}`\n")
    lines.append(f"- Main segment types: `{', '.join(report_segments)}`\n")
    lines.append("\n### Channel completeness\n")
    lines.append(md_table(completeness, ["segment_type", "complete_triplets", "expected_recordings", "incomplete_recordings"]))

    raw_geometry = base.groupby(["segment_type", "channel"], as_index=False).agg(
        n_recordings=("recording_id", "nunique"),
        raw_abs_mean_median=("raw_mean", lambda s: finite_median(pd.Series(s).abs())),
        raw_std_median=("raw_std", finite_median),
        raw_std_dispersion=("raw_std", finite_std),
        raw_iqr_median=("raw_iqr", finite_median),
    )
    lines.append("## Raw geometry\n")
    lines.append("The raw geometry table is the baseline for interpreting the movement plots. It summarizes where recording clusters start before normalization.\n")
    lines.append(md_table(raw_geometry, ["segment_type", "channel", "n_recordings", "raw_abs_mean_median", "raw_std_median", "raw_std_dispersion", "raw_iqr_median"], max_rows=20))

    lines.append("## Normalization shifts\n")
    lines.append("Lower `std_dispersion_ratio_median` means that the spread of recording-level std values became tighter after normalization. `channel_std_cv_ratio` shows whether inter-channel scale differences were preserved or collapsed.\n")
    if not ordered_overall.empty:
        lines.append(md_table(
            ordered_overall,
            [
                "segment_type",
                "normalization_id",
                "std_dispersion_ratio_median",
                "iqr_dispersion_ratio_median",
                "rms_dispersion_ratio_median",
                "channel_std_cv_ratio",
                "norm_channel_std_cv_median",
                "interpretation",
            ],
            max_rows=30,
        ))

    lines.append("## Per-channel cluster movement\n")
    lines.append("This table keeps channels separate. It is useful for detecting methods that work for frontal channels but behave differently on the third OcR/Hipp slot.\n")
    ns = norm_summary[norm_summary["normalization_id"] != "none"].copy()
    ns = ns.sort_values(["segment_type", "normalization_id", "channel"])
    lines.append(md_table(ns, ["segment_type", "normalization_id", "channel", "raw_centroid_std", "norm_centroid_std", "std_dispersion_ratio", "iqr_dispersion_ratio", "norm_abs_mean_median"], max_rows=60))

    lines.append("## Inter-channel balance\n")
    lines.append("The channel-balance layer checks whether a method merely compresses all channels independently or preserves the relative channel scale geometry.\n")
    bs = bal_summary[bal_summary["normalization_id"] != "none"].sort_values(["segment_type", "normalization_id"])
    lines.append(md_table(bs, ["segment_type", "normalization_id", "n_recordings", "raw_channel_std_cv_median", "norm_channel_std_cv_median", "channel_std_cv_ratio", "log_std_ratio_dispersion_ratio"], max_rows=30))

    if not outliers.empty:
        lines.append("## Residual outliers after normalization\n")
        lines.append("These are recordings/channels with the largest residual normalized-std deviations from their method/segment/channel median. They are candidates for manual inspection, not model errors.\n")
        lines.append(md_table(outliers, ["normalization_id", "segment_type", "channel", "recording_id", "norm_std", "deviation"], max_rows=12))

    lines.append("## Figures\n")
    lines.append("Figures are deliberately limited. They support the tables above and show whether cluster movement is visually plausible, rather than replacing the report with a directory dump.\n")
    for rel in rel_figs:
        title = Path(rel).stem.replace("_", " ")
        lines.append(f"\n### {title}\n")
        lines.append(f"![{title}]({rel})\n")

    lines.append("## Output files\n")
    lines.append("- `raw_stats.csv` — cached raw statistics; this is the only expensive layer.\n")
    lines.append("- `normalization_effects.csv` — analytically derived normalized statistics.\n")
    lines.append("- `normalization_summary.csv` — per-normalization/per-channel cluster movement metrics.\n")
    lines.append("- `channel_balance.csv` and `channel_balance_summary.csv` — inter-channel geometry.\n")
    lines.append("- `normalization_overall_summary.csv` — compact table used by the executive summary.\n")
    return "\n".join(lines)


def write_report(
    raw_df: pd.DataFrame,
    effects_df: pd.DataFrame,
    norm_summary: pd.DataFrame,
    bal_df: pd.DataFrame,
    bal_summary: pd.DataFrame,
    overall: pd.DataFrame,
    figure_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path:
    md = report_markdown(raw_df, effects_df, norm_summary, bal_df, bal_summary, overall, figure_paths, out_dir, cfg)
    path = out_dir / "report.md"
    path.write_text(md, encoding="utf-8")
    print(f"[OK] wrote {path}")
    return path


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def run(args: argparse.Namespace) -> None:
    cfg = load_yaml(Path(args.config))
    data_dir = resolve_data_dir(args, cfg)
    out_dir = ensure_dir(Path(args.out).expanduser().resolve())
    raw_path = out_dir / "raw_stats.csv"

    if args.recompute_raw or not raw_path.exists():
        raw_df = compute_raw_stats(data_dir, out_dir, cfg)
    else:
        print(f"[REUSE] reading {raw_path}")
        raw_df = pd.read_csv(raw_path)

    effects_df = derive_effects(raw_df, out_dir, cfg)
    norm_summary = summarize_effects(effects_df, out_dir)
    bal_df, bal_summary = channel_balance(effects_df, out_dir)
    overall = overall_summary(norm_summary, bal_summary, out_dir, cfg)
    figure_paths = make_figures(effects_df, bal_df, overall, out_dir, cfg)
    write_report(raw_df, effects_df, norm_summary, bal_df, bal_summary, overall, figure_paths, out_dir, cfg)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit how normalizations move recording-level EEG statistics.")
    p.add_argument("--config", default="configs/normalization_audit.yaml", help="YAML config path")
    p.add_argument("--data_dir", default=None, help="Override processed data directory")
    p.add_argument("--out", default="reports/normalization_audit", help="Output directory")
    p.add_argument("--recompute_raw", action="store_true", help="Re-read .npy files and recompute raw_stats.csv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()
