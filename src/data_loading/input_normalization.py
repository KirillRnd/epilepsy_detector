from __future__ import annotations

from typing import Any

import numpy as np


SUPPORTED_INPUT_NORMALIZATIONS = (
    "none",
    "per_record_joint_zscore",
    "per_record_per_channel_zscore",
    "per_record_joint_robust",
    "per_record_per_channel_robust",
)

MAD_SCALE = 1.4826
CHANNEL_NAMES = ("FrL", "FrR", "OcR_Hipp")


def _validate_signal(x: np.ndarray) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"Expected input signal with shape (C, T), got {arr.shape}")
    return arr


def _safe_scale(scale: np.ndarray, eps: float) -> tuple[np.ndarray, list[str]]:
    safe = np.asarray(scale, dtype=np.float64).copy()
    bad = (~np.isfinite(safe)) | (np.abs(safe) < eps)
    warnings = ["scale_fallback" if flag else "" for flag in np.ravel(bad)]
    safe[bad] = 1.0
    return safe, warnings


def _stats(
    raw: np.ndarray,
    norm: np.ndarray,
    name: str,
    center: np.ndarray,
    scale: np.ndarray,
    warnings: list[str],
) -> dict[str, Any]:
    n_channels = int(raw.shape[0])
    center_flat = np.ravel(np.asarray(center, dtype=np.float64))
    scale_flat = np.ravel(np.asarray(scale, dtype=np.float64))

    if center_flat.size == 1:
        center_values = np.repeat(center_flat, n_channels)
    else:
        center_values = center_flat
    if scale_flat.size == 1:
        scale_values = np.repeat(scale_flat, n_channels)
    else:
        scale_values = scale_flat
    if len(warnings) == 1:
        warning_values = warnings * n_channels
    else:
        warning_values = warnings

    rows = []
    for idx in range(n_channels):
        channel = CHANNEL_NAMES[idx] if idx < len(CHANNEL_NAMES) else f"ch{idx}"
        rows.append(
            {
                "input_normalization": name,
                "channel": channel,
                "center": float(center_values[idx]),
                "scale": float(scale_values[idx]),
                "raw_mean": float(np.mean(raw[idx])),
                "raw_std": float(np.std(raw[idx])),
                "norm_mean": float(np.mean(norm[idx])),
                "norm_std": float(np.std(norm[idx])),
                "warning_flags": warning_values[idx],
            }
        )
    return {"input_normalization": name, "rows": rows}


def _stats_from_params(
    raw: np.ndarray,
    name: str,
    center: np.ndarray,
    scale: np.ndarray,
    warnings: list[str],
    eps: float,
) -> dict[str, Any]:
    raw = _validate_signal(raw)
    n_channels = int(raw.shape[0])
    center_flat = np.ravel(np.asarray(center, dtype=np.float64))
    scale_flat = np.ravel(np.asarray(scale, dtype=np.float64))

    if center_flat.size == 1:
        center_values = np.repeat(center_flat, n_channels)
    else:
        center_values = center_flat
    if scale_flat.size == 1:
        scale_values = np.repeat(scale_flat, n_channels)
    else:
        scale_values = scale_flat
    if len(warnings) == 1:
        warning_values = warnings * n_channels
    else:
        warning_values = warnings

    rows = []
    for idx in range(n_channels):
        channel = CHANNEL_NAMES[idx] if idx < len(CHANNEL_NAMES) else f"ch{idx}"
        raw_mean = float(np.mean(raw[idx]))
        raw_std = float(np.std(raw[idx]))
        denom = float(scale_values[idx]) + eps
        rows.append(
            {
                "input_normalization": name,
                "channel": channel,
                "center": float(center_values[idx]),
                "scale": float(scale_values[idx]),
                "raw_mean": raw_mean,
                "raw_std": raw_std,
                "norm_mean": (raw_mean - float(center_values[idx])) / denom,
                "norm_std": raw_std / denom,
                "warning_flags": warning_values[idx],
            }
        )
    return {"input_normalization": name, "rows": rows}


def fit_input_normalization(
    x: np.ndarray,
    name: str,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> dict[str, Any] | tuple[dict[str, Any], dict[str, Any]]:
    raw = _validate_signal(x)
    name = str(name or "none")
    if name not in INPUT_NORMALIZERS:
        valid = ", ".join(SUPPORTED_INPUT_NORMALIZATIONS)
        raise ValueError(f"Unknown input_normalization '{name}'. Valid values: {valid}")

    if name == "none":
        center = np.zeros((1, 1), dtype=np.float32)
        scale = np.ones((1, 1), dtype=np.float32)
        warnings = [""]
    elif name == "per_record_joint_zscore":
        center = np.array([[np.mean(raw, dtype=np.float64)]], dtype=np.float64)
        scale, warnings = _safe_scale(
            np.array([[np.std(raw, dtype=np.float64)]], dtype=np.float64),
            eps,
        )
    elif name == "per_record_per_channel_zscore":
        center = np.mean(raw, axis=1, keepdims=True, dtype=np.float64)
        scale, warnings = _safe_scale(
            np.std(raw, axis=1, keepdims=True, dtype=np.float64),
            eps,
        )
    elif name == "per_record_joint_robust":
        center = np.array([[np.median(raw)]], dtype=np.float64)
        mad = np.array([[np.median(np.abs(raw - center[0, 0]))]], dtype=np.float64)
        scale, warnings = _safe_scale(MAD_SCALE * mad, eps)
    elif name == "per_record_per_channel_robust":
        center = np.median(raw, axis=1, keepdims=True)
        mad = np.median(np.abs(raw - center), axis=1, keepdims=True)
        scale, warnings = _safe_scale(MAD_SCALE * mad, eps)

    params = {
        "input_normalization": name,
        "center": np.asarray(center, dtype=np.float32),
        "scale": np.asarray(scale, dtype=np.float32),
        "warnings": warnings,
    }
    if not return_stats:
        return params
    return params, _stats_from_params(raw, name, center, scale, warnings, eps)


def apply_precomputed_input_normalization(
    x: np.ndarray,
    params: dict[str, Any],
    *,
    eps: float = 1e-8,
) -> np.ndarray:
    name = str(params.get("input_normalization", "none"))
    out = np.asarray(x, dtype=np.float32).copy()
    if name == "none":
        return out
    center = np.asarray(params["center"], dtype=np.float32)
    scale = np.asarray(params["scale"], dtype=np.float32)
    return ((out - center) / (scale + eps)).astype(np.float32)


def normalize_none(
    x: np.ndarray,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    raw = _validate_signal(x)
    norm = np.asarray(raw, dtype=np.float32)
    if not return_stats:
        return norm
    stats = _stats(raw, norm, "none", np.array([0.0]), np.array([1.0]), [""])
    return norm, stats


def normalize_per_record_joint_zscore(
    x: np.ndarray,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    raw = _validate_signal(x)
    center = np.array([np.mean(raw, dtype=np.float64)], dtype=np.float64)
    scale, warnings = _safe_scale(np.array([np.std(raw, dtype=np.float64)]), eps)
    norm = ((raw.astype(np.float32, copy=True) - center.astype(np.float32)) / (scale.astype(np.float32) + eps)).astype(np.float32)
    if not return_stats:
        return norm
    return norm, _stats(raw, norm, "per_record_joint_zscore", center, scale, warnings)


def normalize_per_record_per_channel_zscore(
    x: np.ndarray,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    raw = _validate_signal(x)
    center = np.mean(raw, axis=1, keepdims=True, dtype=np.float64)
    scale, warnings = _safe_scale(np.std(raw, axis=1, keepdims=True, dtype=np.float64), eps)
    norm = ((raw.astype(np.float32, copy=True) - center.astype(np.float32)) / (scale.astype(np.float32) + eps)).astype(np.float32)
    if not return_stats:
        return norm
    return norm, _stats(raw, norm, "per_record_per_channel_zscore", center, scale, warnings)


def normalize_per_record_joint_robust(
    x: np.ndarray,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    raw = _validate_signal(x)
    center = np.array([np.median(raw)], dtype=np.float64)
    mad = np.array([np.median(np.abs(raw - center[0]))], dtype=np.float64)
    scale, warnings = _safe_scale(MAD_SCALE * mad, eps)
    norm = ((raw.astype(np.float32, copy=True) - center.astype(np.float32)) / (scale.astype(np.float32) + eps)).astype(np.float32)
    if not return_stats:
        return norm
    return norm, _stats(raw, norm, "per_record_joint_robust", center, scale, warnings)


def normalize_per_record_per_channel_robust(
    x: np.ndarray,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    raw = _validate_signal(x)
    center = np.median(raw, axis=1, keepdims=True)
    mad = np.median(np.abs(raw - center), axis=1, keepdims=True)
    scale, warnings = _safe_scale(MAD_SCALE * mad, eps)
    norm = ((raw.astype(np.float32, copy=True) - center.astype(np.float32)) / (scale.astype(np.float32) + eps)).astype(np.float32)
    if not return_stats:
        return norm
    return norm, _stats(raw, norm, "per_record_per_channel_robust", center, scale, warnings)


INPUT_NORMALIZERS = {
    "none": normalize_none,
    "per_record_joint_zscore": normalize_per_record_joint_zscore,
    "per_record_per_channel_zscore": normalize_per_record_per_channel_zscore,
    "per_record_joint_robust": normalize_per_record_joint_robust,
    "per_record_per_channel_robust": normalize_per_record_per_channel_robust,
}


def apply_input_normalization(
    x: np.ndarray,
    name: str,
    *,
    eps: float = 1e-8,
    return_stats: bool = False,
) -> np.ndarray | tuple[np.ndarray, dict[str, Any]]:
    name = str(name or "none")
    if name not in INPUT_NORMALIZERS:
        valid = ", ".join(SUPPORTED_INPUT_NORMALIZATIONS)
        raise ValueError(f"Unknown input_normalization '{name}'. Valid values: {valid}")
    return INPUT_NORMALIZERS[name](x, eps=eps, return_stats=return_stats)


def input_normalization_from_config(config: dict[str, Any] | None) -> str:
    if not config:
        return "none"
    data_cfg = config.get("data", {}) if isinstance(config.get("data", {}), dict) else {}
    return str(config.get("input_normalization", data_cfg.get("input_normalization", "none")))
