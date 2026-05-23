#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Data-level channel order audit for processed rat EEG/ECoG recordings.

This script intentionally does NOT run a model, save raw probabilities,
threshold predictions, tune hysteresis, compute frame metrics, compute event
metrics, or rewrite processed_signals.npy.

It checks whether the declared logical channel order

    FrL, FrR, OcR_Hipp

is consistent with compact raw signal fingerprints.  The main use case is a
mechanical lab error where electrodes were plugged into wrong inputs before the
processed repository channel order was created.

Design principles copied from audit_normalization.py:
  * expensive step is done once: compute raw statistics per recording/channel;
  * statistics are split by background / seizure using seizure_mask.npy;
  * output CSV files are the durable result; report.md is the human entry point;
  * channel slots are fixed to three logical channels: FrL, FrR, OcR_Hipp.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
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
EPS_DEFAULT = 1e-12


# -----------------------------------------------------------------------------
# Config / paths
# -----------------------------------------------------------------------------


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


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


def cfg_channels(cfg: dict[str, Any]) -> list[str]:
    return list(deep_get(cfg, ["channel_order_audit", "channels"], LOGICAL_CHANNELS))


def expected_order(cfg: dict[str, Any]) -> tuple[str, ...]:
    return tuple(deep_get(cfg, ["channel_order_audit", "expected_order"], cfg_channels(cfg)))


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


def canonical_channel_name(raw_name: str | None, index: int, channels: list[str]) -> str | None:
    """Map repository channel names to logical channel slots.

    If metadata names are absent or unknown, fall back to positional mapping.
    """
    if raw_name is None:
        return channels[index] if index < len(channels) else None
    key = str(raw_name).strip().lower().replace("-", "").replace("_", "").replace("/", "")
    if key in {"frl", "frontleft"}:
        return "FrL"
    if key in {"frr", "frontright"}:
        return "FrR"
    if key in {"ocr", "hipp", "hip", "ocright", "hippocampus", "ocrhipp"}:
        return "OcR_Hipp"
    return channels[index] if index < len(channels) else None


def logical_channel_indices(arr: np.ndarray, metadata: dict[str, Any], channels: list[str]) -> dict[str, int]:
    raw_names = metadata.get("channel_names") or []
    mapping: dict[str, int] = {}
    n_channels = int(arr.shape[0])
    for idx in range(n_channels):
        raw_name = raw_names[idx] if idx < len(raw_names) else None
        logical = canonical_channel_name(raw_name, idx, channels=channels)
        if logical is None:
            continue
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
            "p01": math.nan,
            "p05": math.nan,
            "p25": math.nan,
            "p75": math.nan,
            "p95": math.nan,
            "p99": math.nan,
            "iqr": math.nan,
        }
    p01, p05, p25, med, p75, p95, p99 = np.percentile(v, [1, 5, 25, 50, 75, 95, 99])
    mad = float(np.median(np.abs(v - med)))
    return {
        "median": float(med),
        "mad": mad,
        "mad_scaled": float(mad * mad_scale),
        "p01": float(p01),
        "p05": float(p05),
        "p25": float(p25),
        "p75": float(p75),
        "p95": float(p95),
        "p99": float(p99),
        "iqr": float(p75 - p25),
    }


def segment_moments_from_all_and_seizure(
    x: np.ndarray,
    mask: np.ndarray,
    segment_type: str,
) -> tuple[int, float, float, float, float, float, float, float, float]:
    """Exact mean/std/rms components for all/background/seizure.

    Returns:
        n, sum, sumsq, min, max, finite_ratio, nan_ratio, inf_ratio,
        segment_positive_fraction.

    For background, sums are derived from all - seizure to avoid copying the
    whole background vector.
    """
    n_total = int(x.shape[0])
    finite_all = np.isfinite(x)
    finite_count_all = int(np.count_nonzero(finite_all))
    nan_count_all = int(np.count_nonzero(np.isnan(x)))
    inf_count_all = int(np.count_nonzero(np.isinf(x)))

    if segment_type == "all":
        n, s, ss, mn, mx = finite_moments(x)
        denom = max(1, n_total)
        return n, s, ss, mn, mx, finite_count_all / denom, nan_count_all / denom, inf_count_all / denom, float(np.mean(mask))

    seiz_x = x[mask]
    n_seiz_total = int(seiz_x.shape[0])
    n_seiz, s_seiz, ss_seiz, mn_seiz, mx_seiz = finite_moments(seiz_x)

    if segment_type == "seizure":
        denom = max(1, n_seiz_total)
        return (
            n_seiz,
            s_seiz,
            ss_seiz,
            mn_seiz,
            mx_seiz,
            float(np.isfinite(seiz_x).sum() / denom),
            float(np.isnan(seiz_x).sum() / denom),
            float(np.isinf(seiz_x).sum() / denom),
            1.0 if n_seiz_total else 0.0,
        )

    if segment_type == "background":
        n_all, s_all, ss_all, _mn_all, _mx_all = finite_moments(x)
        n_bg = n_all - n_seiz
        s_bg = s_all - s_seiz
        ss_bg = ss_all - ss_seiz
        n_bg_total = n_total - n_seiz_total
        # Exact min/max for background requires materializing only background finite values.
        if n_bg_total > 0:
            bg_x = x[~mask]
            _n, _s, _ss, mn_bg, mx_bg = finite_moments(bg_x)
        else:
            mn_bg, mx_bg = math.nan, math.nan
        denom = max(1, n_bg_total)
        finite_bg = max(0, finite_count_all - int(np.isfinite(seiz_x).sum()))
        nan_bg = max(0, nan_count_all - int(np.isnan(seiz_x).sum()))
        inf_bg = max(0, inf_count_all - int(np.isinf(seiz_x).sum()))
        return n_bg, s_bg, ss_bg, mn_bg, mx_bg, finite_bg / denom, nan_bg / denom, inf_bg / denom, 0.0

    raise ValueError(f"unknown segment_type: {segment_type}")


def compute_raw_stats_for_record(rec: Recording, cfg: dict[str, Any], segments: list[str]) -> list[dict[str, Any]]:
    channels = cfg_channels(cfg)
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
    channel_map = logical_channel_indices(arr, metadata, channels=channels)
    sfreq = metadata.get("sampling_freq", metadata.get("sfreq", math.nan))
    duration_s = float(metadata.get("duration", n_samples / sfreq if sfreq and np.isfinite(sfreq) else math.nan))

    max_samples = int(deep_get(cfg, ["stats", "robust_sample_max"], 200_000))
    mad_scale = float(deep_get(cfg, ["normalization", "mad_scale"], 1.4826))

    rows: list[dict[str, Any]] = []
    for logical_ch in channels:
        if logical_ch not in channel_map:
            continue
        ch_idx = channel_map[logical_ch]
        x = np.asarray(arr[ch_idx, :n_samples], dtype=np.float64)

        for segment_type in segments:
            rng = np.random.default_rng(stable_seed(rec.recording_id, logical_ch, segment_type))
            n, s, ss, mn, mx, finite_ratio, nan_ratio, inf_ratio, segment_positive_fraction = segment_moments_from_all_and_seizure(
                x, mask, segment_type
            )
            mean, std, rms = stats_from_moments(n, s, ss)
            sample_idx = sample_indices_for_segment(n_samples, mask, segment_type, max_samples, rng)
            sample_values = x[sample_idx] if sample_idx.size else np.asarray([], dtype=np.float64)
            robust = robust_sample_stats(sample_values, mad_scale=mad_scale)

            rows.append(
                {
                    "recording_id": rec.recording_id,
                    "animal_id": rec.animal_id,
                    "session_id": rec.session_id,
                    "segment_type": segment_type,
                    "declared_channel": logical_ch,
                    "channel": logical_ch,  # compatibility with normalization raw_stats.csv
                    "channel_index": int(ch_idx),
                    "n_samples": int(n),
                    "duration_s": duration_s,
                    "sampling_freq": sfreq,
                    "seizure_fraction": float(np.mean(mask)),
                    "mean": mean,
                    "std": std,
                    "rms": rms,
                    "min": mn,
                    "max": mx,
                    "median": robust["median"],
                    "mad": robust["mad"],
                    "mad_scaled": robust["mad_scaled"],
                    "p01": robust["p01"],
                    "p05": robust["p05"],
                    "p25": robust["p25"],
                    "p75": robust["p75"],
                    "p95": robust["p95"],
                    "p99": robust["p99"],
                    "iqr": robust["iqr"],
                    "finite_ratio": finite_ratio,
                    "nan_ratio": nan_ratio,
                    "inf_ratio": inf_ratio,
                    "segment_positive_fraction": segment_positive_fraction,
                    "sampled_robust_stats": bool(n > max_samples),
                }
            )
    return rows


def normalize_raw_stats_columns(df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    """Accept this script's raw CSV or audit_normalization.py raw_stats.csv."""
    channels = cfg_channels(cfg)
    out = df.copy()
    if "declared_channel" not in out.columns and "channel" in out.columns:
        out["declared_channel"] = out["channel"]
    if "channel" not in out.columns and "declared_channel" in out.columns:
        out["channel"] = out["declared_channel"]
    if "is_joint" in out.columns:
        out = out[~out["is_joint"].fillna(False).astype(bool)].copy()
    out = out[out["declared_channel"].isin(channels)].copy()
    report_segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    out = out[out["segment_type"].isin(report_segments)].copy()
    return out


def compute_raw_stats(data_dir: Path, out_dir: Path, cfg: dict[str, Any]) -> pd.DataFrame:
    segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    records = find_recordings(data_dir)
    print(f"Found {len(records)} recordings in {data_dir}")
    all_rows: list[dict[str, Any]] = []
    for i, rec in enumerate(records, start=1):
        print(f"[{i}/{len(records)}] {rec.recording_id}")
        try:
            all_rows.extend(compute_raw_stats_for_record(rec, cfg, segments=segments))
        except Exception as e:
            print(f"[WARN] failed {rec.recording_id}: {e}")

    raw_df = pd.DataFrame(all_rows)
    raw_df = normalize_raw_stats_columns(raw_df, cfg)
    raw_path = out_dir / "raw_channel_stats.csv"
    raw_df.to_csv(raw_path, index=False)
    print(f"[OK] wrote {raw_path}")
    return raw_df


# -----------------------------------------------------------------------------
# Fingerprint features and reference
# -----------------------------------------------------------------------------


def safe_log(x: Any) -> float:
    try:
        v = float(x)
    except Exception:
        return math.nan
    if not np.isfinite(v) or v <= 0:
        return math.nan
    return float(np.log(v))


def add_fingerprint_features(raw_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    out = raw_df.copy()
    for base in ["std", "rms", "iqr", "mad_scaled", "mean", "median"]:
        if base not in out.columns:
            out[base] = math.nan
    out["log_std"] = out["std"].map(safe_log)
    out["log_rms"] = out["rms"].map(safe_log)
    out["log_iqr"] = out["iqr"].map(safe_log)
    out["log_mad_scaled"] = out["mad_scaled"].map(safe_log)
    out["abs_mean"] = pd.to_numeric(out["mean"], errors="coerce").abs()
    out["log_abs_mean"] = out["abs_mean"].map(safe_log)
    return out


def finite_values(s: pd.Series) -> np.ndarray:
    return pd.to_numeric(s, errors="coerce").replace([np.inf, -np.inf], np.nan).dropna().to_numpy(dtype=float)


def robust_center_scale(values: np.ndarray, min_scale: float) -> tuple[float, float, int, float, float]:
    v = np.asarray(values, dtype=float)
    v = v[np.isfinite(v)]
    if v.size == 0:
        return math.nan, math.nan, 0, math.nan, math.nan
    p25, med, p75 = np.percentile(v, [25, 50, 75])
    iqr = float(p75 - p25)
    scale = max(iqr, float(min_scale))
    return float(med), float(scale), int(v.size), float(p25), float(p75)


def build_reference(raw_feat_df: pd.DataFrame, cfg: dict[str, Any], exclude_recordings: set[str] | None = None) -> pd.DataFrame:
    channels = cfg_channels(cfg)
    features = list(deep_get(cfg, ["channel_order_audit", "features"], ["log_std", "log_rms", "log_iqr", "log_mad_scaled"]))
    segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    min_scale = float(deep_get(cfg, ["channel_order_audit", "reference", "min_feature_scale"], 0.05))
    min_records = int(deep_get(cfg, ["channel_order_audit", "reference", "min_records_per_channel_segment"], 10))
    exclude_recordings = exclude_recordings or set()

    df = raw_feat_df[~raw_feat_df["recording_id"].isin(exclude_recordings)].copy()
    rows: list[dict[str, Any]] = []
    for segment in segments:
        for channel in channels:
            g = df[(df["segment_type"] == segment) & (df["declared_channel"] == channel)]
            for feature in features:
                vals = finite_values(g[feature]) if feature in g.columns else np.asarray([], dtype=float)
                center, scale, n, p25, p75 = robust_center_scale(vals, min_scale=min_scale)
                rows.append(
                    {
                        "segment_type": segment,
                        "channel": channel,
                        "feature": feature,
                        "center": center,
                        "scale": scale,
                        "n_recordings": n,
                        "p25": p25,
                        "p75": p75,
                        "min_records_required": min_records,
                        "reference_status": "ok" if n >= min_records and np.isfinite(center) else "low_n_or_missing",
                        "excluded_recordings": len(exclude_recordings),
                    }
                )
    return pd.DataFrame(rows)


def score_permutations(raw_feat_df: pd.DataFrame, reference_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    channels = cfg_channels(cfg)
    features = list(deep_get(cfg, ["channel_order_audit", "features"], ["log_std", "log_rms", "log_iqr", "log_mad_scaled"]))
    segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    expected = expected_order(cfg)
    permutations = list(itertools.permutations(channels))

    ref = reference_df.set_index(["segment_type", "channel", "feature"], drop=False)
    rows: list[dict[str, Any]] = []

    for (recording_id, segment), g in raw_feat_df.groupby(["recording_id", "segment_type"], sort=True):
        if segment not in segments:
            continue
        obs = g.set_index("declared_channel", drop=False)
        animal_id = str(g["animal_id"].iloc[0]) if "animal_id" in g.columns and not g.empty else ""
        session_id = str(g["session_id"].iloc[0]) if "session_id" in g.columns and not g.empty else ""

        for perm in permutations:
            distances: list[float] = []
            n_missing = 0
            for declared_ch, assigned_actual_ch in zip(channels, perm):
                if declared_ch not in obs.index:
                    n_missing += len(features)
                    continue
                obs_row = obs.loc[declared_ch]
                if isinstance(obs_row, pd.DataFrame):
                    obs_row = obs_row.iloc[0]
                for feature in features:
                    if feature not in obs_row.index:
                        n_missing += 1
                        continue
                    val = obs_row[feature]
                    key = (segment, assigned_actual_ch, feature)
                    if key not in ref.index:
                        n_missing += 1
                        continue
                    ref_row = ref.loc[key]
                    if isinstance(ref_row, pd.DataFrame):
                        ref_row = ref_row.iloc[0]
                    center = float(ref_row["center"])
                    scale = float(ref_row["scale"])
                    try:
                        v = float(val)
                    except Exception:
                        n_missing += 1
                        continue
                    if not np.isfinite(v) or not np.isfinite(center) or not np.isfinite(scale) or scale <= 0:
                        n_missing += 1
                        continue
                    distances.append(abs(v - center) / scale)

            score = float(np.mean(distances)) if distances else math.nan
            rows.append(
                {
                    "recording_id": recording_id,
                    "animal_id": animal_id,
                    "session_id": session_id,
                    "segment_type": segment,
                    "permutation": ",".join(perm),
                    "is_expected": tuple(perm) == expected,
                    "score": score,
                    "n_terms": int(len(distances)),
                    "n_missing_terms": int(n_missing),
                }
            )

    scores = pd.DataFrame(rows)
    if scores.empty:
        return scores

    # Add per-recording/per-segment rank and margin against expected.
    out_rows: list[pd.DataFrame] = []
    for (_rec, _seg), g in scores.groupby(["recording_id", "segment_type"], sort=False):
        tmp = g.copy()
        tmp = tmp.sort_values("score", ascending=True, na_position="last").reset_index(drop=True)
        tmp["rank"] = np.arange(1, len(tmp) + 1)
        best_perm = str(tmp.iloc[0]["permutation"]) if not tmp.empty else ""
        best_score = float(tmp.iloc[0]["score"]) if not tmp.empty else math.nan
        expected_rows = tmp[tmp["is_expected"]]
        expected_score = float(expected_rows["score"].iloc[0]) if not expected_rows.empty else math.nan
        tmp["best_permutation_for_segment"] = best_perm
        tmp["best_score_for_segment"] = best_score
        tmp["expected_score_for_segment"] = expected_score
        tmp["score_margin_vs_expected"] = expected_score - tmp["score"] if np.isfinite(expected_score) else math.nan
        out_rows.append(tmp)
    return pd.concat(out_rows, ignore_index=True) if out_rows else scores


def build_reference_iterative(raw_feat_df: pd.DataFrame, cfg: dict[str, Any]) -> tuple[pd.DataFrame, pd.DataFrame, set[str]]:
    """Build reference; optionally exclude strong suspicious records and rebuild once."""
    do_iter = bool(deep_get(cfg, ["channel_order_audit", "reference", "exclude_suspicious_iteratively"], False))
    max_passes = int(deep_get(cfg, ["channel_order_audit", "reference", "max_reference_passes"], 2))
    exclude_margin = float(deep_get(cfg, ["channel_order_audit", "reference", "exclude_margin"], 3.0))
    primary = str(deep_get(cfg, ["channel_order_audit", "primary_segment"], "background"))
    expected = ",".join(expected_order(cfg))

    excluded: set[str] = set()
    reference_df = pd.DataFrame()
    scores_df = pd.DataFrame()

    for pass_idx in range(max(1, max_passes)):
        reference_df = build_reference(raw_feat_df, cfg, exclude_recordings=excluded)
        scores_df = score_permutations(raw_feat_df, reference_df, cfg)
        if not do_iter or pass_idx >= max_passes - 1 or scores_df.empty:
            break
        best = scores_df[(scores_df["segment_type"] == primary) & (scores_df["rank"] == 1)].copy()
        new_excluded = set(
            best[(best["permutation"] != expected) & (best["score_margin_vs_expected"] >= exclude_margin)]["recording_id"].astype(str)
        )
        if not new_excluded or new_excluded.issubset(excluded):
            break
        excluded.update(new_excluded)
        print(f"[INFO] reference pass {pass_idx + 1}: excluding {len(new_excluded)} strong suspicious recordings")

    return reference_df, scores_df, excluded


# -----------------------------------------------------------------------------
# Simple rank checks and decisions
# -----------------------------------------------------------------------------


def descending_std_order(g: pd.DataFrame, channels: list[str]) -> str:
    rows = []
    for ch in channels:
        r = g[g["declared_channel"] == ch]
        if r.empty:
            continue
        v = pd.to_numeric(r["std"], errors="coerce").replace([np.inf, -np.inf], np.nan).dropna()
        if v.empty:
            continue
        rows.append((ch, float(v.iloc[0])))
    if not rows:
        return ""
    rows = sorted(rows, key=lambda x: x[1], reverse=True)
    return ">".join(ch for ch, _ in rows)


def rank_order_table(raw_feat_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    channels = cfg_channels(cfg)
    expected_rank = list(deep_get(cfg, ["channel_order_audit", "simple_rank_check", "expected_descending_std_order"], channels))
    expected_rank_str = ">".join(expected_rank)
    rows: list[dict[str, Any]] = []
    for (rec, seg), g in raw_feat_df.groupby(["recording_id", "segment_type"], sort=True):
        animal_id = str(g["animal_id"].iloc[0]) if "animal_id" in g.columns and not g.empty else ""
        session_id = str(g["session_id"].iloc[0]) if "session_id" in g.columns and not g.empty else ""
        order = descending_std_order(g, channels)
        rows.append(
            {
                "recording_id": rec,
                "animal_id": animal_id,
                "session_id": session_id,
                "segment_type": seg,
                "raw_std_descending_order": order,
                "expected_raw_std_descending_order": expected_rank_str,
                "rank_violation": bool(order and order != expected_rank_str),
            }
        )
    return pd.DataFrame(rows)


def permutation_to_decision(best_perm: str, expected_perm: str) -> str:
    if not best_perm or best_perm == expected_perm:
        return "ok"
    bp = best_perm.split(",")
    ep = expected_perm.split(",")
    if sorted(bp) != sorted(ep) or len(bp) != 3:
        return "likely_complex_permutation"
    diffs = [i for i, (a, b) in enumerate(zip(bp, ep)) if a != b]
    if len(diffs) == 2:
        swapped_expected = sorted([ep[diffs[0]], ep[diffs[1]]])
        if swapped_expected == sorted(["FrL", "FrR"]):
            return "likely_FrL_FrR_swap"
        if swapped_expected == sorted(["FrL", "OcR_Hipp"]):
            return "likely_FrL_OcR_Hipp_swap"
        if swapped_expected == sorted(["FrR", "OcR_Hipp"]):
            return "likely_FrR_OcR_Hipp_swap"
    return "likely_complex_permutation"


def make_decisions(scores_df: pd.DataFrame, rank_df: pd.DataFrame, cfg: dict[str, Any]) -> pd.DataFrame:
    primary = str(deep_get(cfg, ["channel_order_audit", "primary_segment"], "background"))
    secondary = str(deep_get(cfg, ["channel_order_audit", "secondary_segment"], "seizure"))
    expected_perm = ",".join(expected_order(cfg))
    weak_margin = float(deep_get(cfg, ["channel_order_audit", "decisions", "weak_margin"], 1.0))
    strong_margin = float(deep_get(cfg, ["channel_order_audit", "decisions", "strong_margin"], 2.0))
    very_strong_margin = float(deep_get(cfg, ["channel_order_audit", "decisions", "very_strong_margin"], 3.0))
    require_secondary = bool(deep_get(cfg, ["channel_order_audit", "decisions", "require_secondary_agreement_for_likely"], True))
    secondary_min_margin = float(deep_get(cfg, ["channel_order_audit", "decisions", "secondary_agreement_min_margin"], 1.0))
    bad_recording_score = float(deep_get(cfg, ["channel_order_audit", "decisions", "bad_recording_score"], 8.0))

    best = scores_df[scores_df["rank"] == 1].copy() if not scores_df.empty else pd.DataFrame()
    exp_rows = scores_df[scores_df["is_expected"]].copy() if not scores_df.empty else pd.DataFrame()
    best_idx = best.set_index(["recording_id", "segment_type"], drop=False)
    exp_idx = exp_rows.set_index(["recording_id", "segment_type"], drop=False)
    rank_idx = rank_df.set_index(["recording_id", "segment_type"], drop=False) if not rank_df.empty else pd.DataFrame()

    rec_ids = sorted(set(scores_df["recording_id"].astype(str).tolist())) if not scores_df.empty else []
    rows: list[dict[str, Any]] = []
    for rec in rec_ids:
        animal_id = ""
        session_id = ""
        r0 = scores_df[scores_df["recording_id"] == rec].head(1)
        if not r0.empty:
            animal_id = str(r0["animal_id"].iloc[0])
            session_id = str(r0["session_id"].iloc[0])

        def get_best(seg: str) -> tuple[str, float, float, float, int]:
            key = (rec, seg)
            if key not in best_idx.index:
                return "", math.nan, math.nan, math.nan, 0
            b = best_idx.loc[key]
            if isinstance(b, pd.DataFrame):
                b = b.iloc[0]
            exp_score = math.nan
            if key in exp_idx.index:
                e = exp_idx.loc[key]
                if isinstance(e, pd.DataFrame):
                    e = e.iloc[0]
                exp_score = float(e["score"])
            best_score = float(b["score"])
            margin = exp_score - best_score if np.isfinite(exp_score) and np.isfinite(best_score) else math.nan
            return str(b["permutation"]), best_score, exp_score, margin, int(b.get("n_terms", 0))

        best_primary, best_score_primary, expected_score_primary, margin_primary, n_terms_primary = get_best(primary)
        best_secondary, best_score_secondary, expected_score_secondary, margin_secondary, n_terms_secondary = get_best(secondary)

        def get_rank(seg: str) -> tuple[str, bool]:
            key = (rec, seg)
            if isinstance(rank_idx, pd.DataFrame) and not rank_idx.empty and key in rank_idx.index:
                rr = rank_idx.loc[key]
                if isinstance(rr, pd.DataFrame):
                    rr = rr.iloc[0]
                return str(rr["raw_std_descending_order"]), bool(rr["rank_violation"])
            return "", False

        rank_primary, rank_violation_primary = get_rank(primary)
        rank_secondary, rank_violation_secondary = get_rank(secondary)

        secondary_agrees = (
            best_secondary == best_primary
            and best_primary != expected_perm
            and np.isfinite(margin_secondary)
            and margin_secondary >= secondary_min_margin
        )

        decision = "ok"
        confidence = 0.0
        recommended_action = "keep_declared_order"

        if not best_primary or n_terms_primary <= 0 or not np.isfinite(best_score_primary):
            decision = "insufficient_data"
            confidence = 0.0
            recommended_action = "check_raw_files_and_metadata"
        elif best_primary == expected_perm or not np.isfinite(margin_primary) or margin_primary < weak_margin:
            # No better alternative or only tiny gain.
            if best_score_primary >= bad_recording_score:
                decision = "bad_recording_or_domain_shift"
                confidence = 0.3
                recommended_action = "manual_quality_review_not_channel_correction"
            else:
                decision = "ok"
                confidence = 0.0
                recommended_action = "keep_declared_order"
        elif margin_primary >= strong_margin:
            base_decision = permutation_to_decision(best_primary, expected_perm)
            if require_secondary and n_terms_secondary > 0:
                if secondary_agrees:
                    decision = base_decision
                    confidence = 0.9 if margin_primary >= very_strong_margin else 0.8
                    recommended_action = "manual_review_then_consider_channel_order_fix"
                else:
                    decision = "weak_warning"
                    confidence = 0.55
                    recommended_action = "manual_review_background_and_seizure_disagree"
            else:
                decision = base_decision
                confidence = 0.75 if margin_primary >= very_strong_margin else 0.65
                recommended_action = "manual_review_then_consider_channel_order_fix"
        else:
            decision = "weak_warning"
            confidence = 0.4
            recommended_action = "manual_review_do_not_autofix"

        # Simple rank-order violation is allowed to raise ok -> weak_warning, but not
        # to override a strong permutation decision.
        if decision == "ok" and rank_violation_primary:
            decision = "weak_warning"
            confidence = max(confidence, 0.35)
            recommended_action = "manual_review_raw_std_rank_violation"

        rows.append(
            {
                "recording_id": rec,
                "animal_id": animal_id,
                "session_id": session_id,
                "primary_segment": primary,
                "secondary_segment": secondary,
                "expected_permutation": expected_perm,
                "best_permutation_primary": best_primary,
                "expected_score_primary": expected_score_primary,
                "best_score_primary": best_score_primary,
                "margin_primary": margin_primary,
                "n_terms_primary": n_terms_primary,
                "best_permutation_secondary": best_secondary,
                "expected_score_secondary": expected_score_secondary,
                "best_score_secondary": best_score_secondary,
                "margin_secondary": margin_secondary,
                "n_terms_secondary": n_terms_secondary,
                "secondary_agrees_with_primary": secondary_agrees,
                "raw_std_order_primary": rank_primary,
                "rank_violation_primary": rank_violation_primary,
                "raw_std_order_secondary": rank_secondary,
                "rank_violation_secondary": rank_violation_secondary,
                "decision": decision,
                "confidence": confidence,
                "recommended_action": recommended_action,
            }
        )
    return pd.DataFrame(rows)


# -----------------------------------------------------------------------------
# Plots
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


def plot_raw_geometry(raw_feat_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    paths: list[Path] = []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    channels = cfg_channels(cfg)
    colors = animal_colors(raw_feat_df["animal_id"])
    for seg in deep_get(cfg, ["segments", "report"], ["background", "seizure"]):
        fig, axes = plt.subplots(1, len(channels), figsize=(4.35 * len(channels), 3.4), squeeze=False)
        for j, ch in enumerate(channels):
            ax = axes[0, j]
            g = raw_feat_df[(raw_feat_df["segment_type"] == seg) & (raw_feat_df["declared_channel"] == ch)]
            scatter_by_animal(ax, g, "mean", "std", colors, marker="o")
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


def plot_permutation_margins(decisions_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    if decisions_df.empty:
        return []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    df = decisions_df.sort_values("margin_primary", ascending=False).copy()
    # Keep figure readable.
    df = df.head(60)
    fig, ax = plt.subplots(1, 1, figsize=(11, max(4.0, 0.18 * len(df))))
    y = np.arange(len(df))
    ax.barh(y, df["margin_primary"].astype(float))
    ax.set_yticks(y)
    ax.set_yticklabels(df["recording_id"].astype(str), fontsize=7)
    ax.invert_yaxis()
    weak = float(deep_get(cfg, ["channel_order_audit", "decisions", "weak_margin"], 1.0))
    strong = float(deep_get(cfg, ["channel_order_audit", "decisions", "strong_margin"], 2.0))
    ax.axvline(weak, linestyle="--", linewidth=0.8, color="0.5", label="weak margin")
    ax.axvline(strong, linestyle="--", linewidth=0.8, color="0.25", label="strong margin")
    ax.set_xlabel("expected_score - best_score on primary segment")
    ax.set_title("Largest channel-order permutation margins")
    ax.grid(True, axis="x", alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.38, right=0.98, bottom=0.1, top=0.9)
    path = fig_dir / "02_permutation_score_margins.png"
    safe_save(fig, path, dpi)
    return [path]


def plot_suspicious_fingerprints(raw_feat_df: pd.DataFrame, decisions_df: pd.DataFrame, fig_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    if decisions_df.empty:
        return []
    dpi = int(deep_get(cfg, ["plots", "dpi"], 140))
    channels = cfg_channels(cfg)
    primary = str(deep_get(cfg, ["channel_order_audit", "primary_segment"], "background"))
    suspicious = decisions_df[decisions_df["decision"].astype(str) != "ok"].sort_values("margin_primary", ascending=False).head(20)
    if suspicious.empty:
        return []

    fig, ax = plt.subplots(1, 1, figsize=(11, max(4.0, 0.28 * len(suspicious))))
    width = 0.22
    x = np.arange(len(suspicious))
    for j, ch in enumerate(channels):
        vals = []
        for rec in suspicious["recording_id"]:
            r = raw_feat_df[(raw_feat_df["recording_id"] == rec) & (raw_feat_df["segment_type"] == primary) & (raw_feat_df["declared_channel"] == ch)]
            vals.append(float(r["std"].iloc[0]) if not r.empty else math.nan)
        ax.bar(x + (j - 1) * width, vals, width=width, label=ch)
    ax.set_xticks(x)
    ax.set_xticklabels(suspicious["recording_id"].astype(str), rotation=75, ha="right", fontsize=7)
    ax.set_ylabel(f"raw std | {primary}")
    ax.set_title("Raw std fingerprints for non-ok recordings")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False)
    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.38, top=0.88)
    path = fig_dir / "03_suspicious_recordings_fingerprints.png"
    safe_save(fig, path, dpi)
    return [path]


def make_figures(raw_feat_df: pd.DataFrame, decisions_df: pd.DataFrame, out_dir: Path, cfg: dict[str, Any]) -> list[Path]:
    if not bool(deep_get(cfg, ["plots", "enabled"], True)):
        return []
    fig_dir = ensure_dir(out_dir / "figures")
    if bool(deep_get(cfg, ["plots", "clean_main_dir"], True)) and fig_dir.exists():
        shutil.rmtree(fig_dir)
        fig_dir.mkdir(parents=True, exist_ok=True)
    paths: list[Path] = []
    paths.extend(plot_raw_geometry(raw_feat_df, fig_dir, cfg))
    paths.extend(plot_permutation_margins(decisions_df, fig_dir, cfg))
    paths.extend(plot_suspicious_fingerprints(raw_feat_df, decisions_df, fig_dir, cfg))
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


def md_table(df: pd.DataFrame, columns: list[str], max_rows: int = 30, digits: int = 4) -> str:
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


def finite_median_series(s: pd.Series) -> float:
    v = finite_values(s)
    return float(np.median(v)) if v.size else math.nan


def report_markdown(
    raw_feat_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    excluded_reference_records: set[str],
    figure_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
) -> str:
    channels = cfg_channels(cfg)
    segments = list(deep_get(cfg, ["segments", "report"], ["background", "seizure"]))
    primary = str(deep_get(cfg, ["channel_order_audit", "primary_segment"], "background"))
    secondary = str(deep_get(cfg, ["channel_order_audit", "secondary_segment"], "seizure"))
    expected_perm = ",".join(expected_order(cfg))
    n_recordings = int(raw_feat_df["recording_id"].nunique()) if not raw_feat_df.empty else 0

    decision_counts = decisions_df["decision"].value_counts().rename_axis("decision").reset_index(name="n") if not decisions_df.empty else pd.DataFrame(columns=["decision", "n"])
    suspicious = decisions_df[decisions_df["decision"].astype(str) != "ok"].sort_values(["confidence", "margin_primary"], ascending=False) if not decisions_df.empty else pd.DataFrame()

    completeness_rows = []
    for seg in segments:
        g = raw_feat_df[raw_feat_df["segment_type"] == seg]
        counts = g.groupby("recording_id")["declared_channel"].nunique()
        completeness_rows.append(
            {
                "segment_type": seg,
                "complete_triplets": int((counts == len(channels)).sum()),
                "expected_recordings": n_recordings,
                "incomplete_recordings": int((counts < len(channels)).sum()),
            }
        )
    completeness = pd.DataFrame(completeness_rows)

    raw_geometry = raw_feat_df.groupby(["segment_type", "declared_channel"], as_index=False).agg(
        n_recordings=("recording_id", "nunique"),
        raw_abs_mean_median=("mean", lambda s: finite_median_series(pd.Series(s).abs())),
        raw_std_median=("std", finite_median_series),
        raw_rms_median=("rms", finite_median_series),
        raw_iqr_median=("iqr", finite_median_series),
        raw_mad_scaled_median=("mad_scaled", finite_median_series),
    )

    rank_violations = rank_df[rank_df["rank_violation"]].copy() if not rank_df.empty else pd.DataFrame()
    rel_figs = [p.relative_to(out_dir).as_posix() for p in figure_paths]

    lines: list[str] = []
    lines.append("# Channel Order Audit\n")
    lines.append(f"Generated: `{datetime.now().isoformat(timespec='seconds')}`\n")

    lines.append("## Scope\n")
    lines.append(
        "This report audits only raw channel-order fingerprints. It does not run model inference, save raw probabilities, threshold outputs, tune hysteresis, compute frame metrics, compute event metrics, or rewrite processed signal files.\n"
    )

    lines.append("## Executive summary\n")
    lines.append(f"- Recordings checked: `{n_recordings}`\n")
    lines.append(f"- Declared / expected channel order: `{expected_perm}`\n")
    lines.append(f"- Primary segment: `{primary}`; secondary segment: `{secondary}`\n")
    lines.append(f"- Reference records excluded during optional robust-reference pass: `{len(excluded_reference_records)}`\n")
    if not decision_counts.empty:
        for _, r in decision_counts.iterrows():
            lines.append(f"- `{r['decision']}`: `{int(r['n'])}`\n")
    if not suspicious.empty:
        lines.append("\nTop manual-review candidates:\n")
        lines.append(md_table(
            suspicious,
            [
                "recording_id",
                "decision",
                "confidence",
                "best_permutation_primary",
                "margin_primary",
                "best_permutation_secondary",
                "margin_secondary",
                "raw_std_order_primary",
                "recommended_action",
            ],
            max_rows=20,
        ))
    else:
        lines.append("\nNo non-ok recordings were found by the current thresholds.\n")

    lines.append("## Data sanity\n")
    lines.append(f"- Logical channels: `{', '.join(channels)}`\n")
    lines.append(f"- Segments: `{', '.join(segments)}`\n")
    lines.append("\n### Channel completeness\n")
    lines.append(md_table(completeness, ["segment_type", "complete_triplets", "expected_recordings", "incomplete_recordings"]))

    lines.append("## Expected raw geometry\n")
    expected_rank = ">".join(deep_get(cfg, ["channel_order_audit", "simple_rank_check", "expected_descending_std_order"], channels))
    lines.append(f"Simple raw-std sanity check expects descending order `{expected_rank}`. This is only a warning layer; the main decision uses robust permutation scores over log-amplitude features.\n")
    lines.append(md_table(raw_geometry, ["segment_type", "declared_channel", "n_recordings", "raw_abs_mean_median", "raw_std_median", "raw_rms_median", "raw_iqr_median", "raw_mad_scaled_median"], max_rows=20))

    lines.append("## Fingerprint reference\n")
    lines.append("Per-channel robust centers/scales used for permutation scoring. Features are usually log-transformed amplitude statistics.\n")
    lines.append(md_table(reference_df, ["segment_type", "channel", "feature", "center", "scale", "n_recordings", "reference_status"], max_rows=80))

    lines.append("## Channel order decisions\n")
    lines.append("Margin means `expected_score - best_score`; positive values mean that another permutation explains the raw fingerprint better than the declared order.\n")
    if not decisions_df.empty:
        ordered = decisions_df.sort_values(["decision", "confidence", "margin_primary"], ascending=[True, False, False])
        lines.append(md_table(
            ordered,
            [
                "recording_id",
                "decision",
                "confidence",
                "best_permutation_primary",
                "expected_score_primary",
                "best_score_primary",
                "margin_primary",
                "best_permutation_secondary",
                "margin_secondary",
                "raw_std_order_primary",
                "recommended_action",
            ],
            max_rows=100,
        ))

    lines.append("## Rank-order violations\n")
    if not rank_violations.empty:
        lines.append(md_table(rank_violations, ["recording_id", "segment_type", "raw_std_descending_order", "expected_raw_std_descending_order"], max_rows=80))
    else:
        lines.append("_No simple raw-std rank violations._\n")

    lines.append("## Figures\n")
    for rel in rel_figs:
        title = Path(rel).stem.replace("_", " ")
        lines.append(f"\n### {title}\n")
        lines.append(f"![{title}]({rel})\n")

    lines.append("## Output files\n")
    lines.append("- `raw_channel_stats.csv` — compact raw statistics per recording/channel/segment.\n")
    lines.append("- `channel_fingerprint_reference.csv` — robust channel fingerprints used as reference.\n")
    lines.append("- `channel_order_permutation_scores.csv` — all six permutation scores per recording/segment.\n")
    lines.append("- `channel_std_rank_order.csv` — simple descending-std sanity checks.\n")
    lines.append("- `channel_order_decisions.csv` — one row per recording with final warning/decision.\n")
    lines.append("- `channel_order_report.md` — this report.\n")

    return "\n".join(lines)


def write_report(
    raw_feat_df: pd.DataFrame,
    reference_df: pd.DataFrame,
    scores_df: pd.DataFrame,
    rank_df: pd.DataFrame,
    decisions_df: pd.DataFrame,
    excluded_reference_records: set[str],
    figure_paths: list[Path],
    out_dir: Path,
    cfg: dict[str, Any],
) -> Path:
    md = report_markdown(raw_feat_df, reference_df, scores_df, rank_df, decisions_df, excluded_reference_records, figure_paths, out_dir, cfg)
    path = out_dir / "channel_order_report.md"
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

    raw_stats_arg = Path(args.raw_stats).expanduser().resolve() if args.raw_stats else None
    raw_path = out_dir / "raw_channel_stats.csv"

    if raw_stats_arg is not None:
        print(f"[REUSE] reading explicit raw stats: {raw_stats_arg}")
        raw_df = pd.read_csv(raw_stats_arg)
        raw_df = normalize_raw_stats_columns(raw_df, cfg)
        raw_df.to_csv(raw_path, index=False)
        print(f"[OK] wrote normalized raw channel stats copy: {raw_path}")
    elif args.recompute_raw or not raw_path.exists():
        raw_df = compute_raw_stats(data_dir, out_dir, cfg)
    else:
        print(f"[REUSE] reading {raw_path}")
        raw_df = pd.read_csv(raw_path)
        raw_df = normalize_raw_stats_columns(raw_df, cfg)

    raw_feat_df = add_fingerprint_features(raw_df, cfg)
    raw_feat_df.to_csv(raw_path, index=False)
    print(f"[OK] wrote {raw_path}")

    reference_df, scores_df, excluded_reference_records = build_reference_iterative(raw_feat_df, cfg)
    ref_path = out_dir / "channel_fingerprint_reference.csv"
    reference_df.to_csv(ref_path, index=False)
    print(f"[OK] wrote {ref_path}")

    scores_path = out_dir / "channel_order_permutation_scores.csv"
    scores_df.to_csv(scores_path, index=False)
    print(f"[OK] wrote {scores_path}")

    rank_df = rank_order_table(raw_feat_df, cfg)
    rank_path = out_dir / "channel_std_rank_order.csv"
    rank_df.to_csv(rank_path, index=False)
    print(f"[OK] wrote {rank_path}")

    decisions_df = make_decisions(scores_df, rank_df, cfg)
    decisions_path = out_dir / "channel_order_decisions.csv"
    decisions_df.to_csv(decisions_path, index=False)
    print(f"[OK] wrote {decisions_path}")

    figure_paths = make_figures(raw_feat_df, decisions_df, out_dir, cfg)
    write_report(raw_feat_df, reference_df, scores_df, rank_df, decisions_df, excluded_reference_records, figure_paths, out_dir, cfg)


def parse_args(argv: list[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Audit whether declared EEG/ECoG channel order matches raw channel fingerprints.")
    p.add_argument("--config", default="configs/channel_order_audit.yaml", help="YAML config path")
    p.add_argument("--data_dir", default=None, help="Override processed data directory")
    p.add_argument("--out", default="reports/channel_order_audit", help="Output directory")
    p.add_argument("--raw_stats", default=None, help="Optional existing raw_stats.csv/raw_channel_stats.csv to reuse")
    p.add_argument("--recompute_raw", action="store_true", help="Re-read .npy files and recompute raw_channel_stats.csv")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> None:
    args = parse_args(sys.argv[1:] if argv is None else argv)
    run(args)


if __name__ == "__main__":
    main()
