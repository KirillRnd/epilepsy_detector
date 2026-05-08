from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class HysteresisParams:
    onset: float = 0.3
    offset: float = 0.15
    min_duration_s: float = 3.0
    min_gap_s: float = 2.0
    collar_s: float = 0.0


@dataclass(frozen=True)
class SegmentMetrics:
    sample_precision: float
    sample_recall: float
    sample_f1: float
    event_precision: float
    event_recall: float
    event_f1: float
    sample_tp: int
    sample_pred: int
    sample_true: int
    event_tp: int
    event_pred: int
    event_true: int


def _empty_segments_array() -> np.ndarray:
    return np.empty((0, 2), dtype=np.int64)


def _segments_array_to_list(segments: np.ndarray) -> list[tuple[int, int]]:
    if segments.size == 0:
        return []
    return [(int(start), int(end)) for start, end in segments]


def _as_segments_array(segments) -> np.ndarray:
    if isinstance(segments, np.ndarray):
        arr = segments
        if arr.size == 0:
            return _empty_segments_array()
        return np.asarray(arr, dtype=np.int64).reshape(-1, 2)

    if not segments:
        return _empty_segments_array()

    arr = np.asarray(segments, dtype=np.int64)
    if arr.size == 0:
        return _empty_segments_array()
    return arr.reshape(-1, 2)


def _merge_segments_array(
    segments,
    min_gap_samples: int = 0,
    assume_sorted: bool = False,
) -> np.ndarray:
    arr = _as_segments_array(segments)
    if arr.size == 0:
        return arr

    starts = arr[:, 0]
    ends = arr[:, 1]
    valid = ends > starts
    if not bool(np.all(valid)):
        arr = arr[valid]
        if arr.size == 0:
            return _empty_segments_array()
        starts = arr[:, 0]
        ends = arr[:, 1]

    if not assume_sorted:
        order = np.argsort(starts, kind="stable")
        arr = arr[order]
        starts = arr[:, 0]
        ends = arr[:, 1]

    if arr.shape[0] == 1:
        return arr.astype(np.int64, copy=False)

    new_group = np.empty(arr.shape[0], dtype=bool)
    new_group[0] = True
    new_group[1:] = (starts[1:] - ends[:-1]) > min_gap_samples
    group_starts = np.flatnonzero(new_group)

    merged_starts = starts[group_starts]
    merged_ends = np.maximum.reduceat(ends, group_starts)
    return np.column_stack((merged_starts, merged_ends)).astype(np.int64, copy=False)


def _merge_sample_segments(
    segments: list[tuple[int, int]],
    min_gap_samples: int = 0,
    assume_sorted: bool = False,
) -> list[tuple[int, int]]:
    return _segments_array_to_list(
        _merge_segments_array(
            segments,
            min_gap_samples=min_gap_samples,
            assume_sorted=assume_sorted,
        )
    )


def hysteresis_raw_segments(
    probs: np.ndarray,
    onset: float = 0.3,
    offset: float = 0.15,
    chunk_size: int = 1_000_000,
) -> np.ndarray:
    """Return raw hysteresis intervals before gap/duration/collar filtering."""
    probs = np.asarray(probs).reshape(-1)
    if probs.size == 0:
        return _empty_segments_array()
    if offset >= onset:
        raise ValueError("offset must be lower than onset for hysteresis")

    n_samples = int(probs.size)
    active = False
    seg_start: int | None = None
    start_chunks: list[np.ndarray] = []
    end_chunks: list[np.ndarray] = []

    chunk_size = int(chunk_size)
    if chunk_size <= 0:
        chunk_size = n_samples

    for chunk_start in range(0, n_samples, chunk_size):
        chunk = probs[chunk_start : chunk_start + chunk_size]
        chunk_len = int(chunk.shape[0])
        if chunk_len == 0:
            continue
        carry_active = active

        state = np.zeros(chunk_len, dtype=np.int8)
        state[chunk >= onset] = 1
        state[chunk < offset] = -1

        marked = state != 0
        positions = np.where(marked, np.arange(chunk_len, dtype=np.int32) + 1, 0)
        last_mark = np.maximum.accumulate(positions) - 1

        active_arr = np.empty(chunk_len, dtype=bool)
        has_mark = last_mark >= 0
        active_arr[has_mark] = state[last_mark[has_mark]] == 1
        active_arr[~has_mark] = active

        if (not carry_active) and bool(active_arr[0]):
            start_chunks.append(np.array([chunk_start], dtype=np.int64))

        if carry_active and not bool(active_arr[0]):
            end_chunks.append(np.array([chunk_start], dtype=np.int64))

        transitions = np.flatnonzero(active_arr[1:] != active_arr[:-1]) + 1
        if transitions.size:
            transition_states = active_arr[transitions]
            starts = transitions[transition_states] + chunk_start
            ends = transitions[~transition_states] + chunk_start
            if starts.size:
                start_chunks.append(starts.astype(np.int64, copy=False))
            if ends.size:
                end_chunks.append(ends.astype(np.int64, copy=False))

        active = bool(active_arr[-1])

    if active:
        end_chunks.append(np.array([n_samples], dtype=np.int64))

    if not start_chunks or not end_chunks:
        return _empty_segments_array()

    starts = np.concatenate(start_chunks)
    ends = np.concatenate(end_chunks)
    n_segments = starts.size if starts.size <= ends.size else ends.size
    if n_segments == 0:
        return _empty_segments_array()
    return np.column_stack((starts[:n_segments], ends[:n_segments])).astype(np.int64, copy=False)


def finalize_hysteresis_segments(
    segments: list[tuple[int, int]],
    n_samples: int,
    sr: float = 400.0,
    min_duration_s: float = 3.0,
    min_gap_s: float = 2.0,
    collar_s: float = 0.0,
) -> list[tuple[int, int]]:
    """Apply min-gap, min-duration and collar to raw hysteresis intervals."""
    segments_arr = _as_segments_array(segments)
    if segments_arr.size == 0:
        return []

    min_dur_samples = int(min_duration_s * sr)
    min_gap_samples = int(min_gap_s * sr)
    collar_samples = int(collar_s * sr)

    merged = _merge_segments_array(
        segments_arr,
        min_gap_samples=min_gap_samples - 1,
        assume_sorted=True,
    )
    if merged.size == 0:
        return []

    durations = merged[:, 1] - merged[:, 0]
    merged = merged[durations >= min_dur_samples]
    if merged.size == 0:
        return []

    if collar_samples > 0:
        merged = merged.copy()
        merged[:, 0] -= collar_samples
        merged[:, 1] += collar_samples
        np.maximum(merged[:, 0], 0, out=merged[:, 0])
        np.minimum(merged[:, 1], int(n_samples), out=merged[:, 1])

    return _segments_array_to_list(merged)


def postprocess_samples(
    probs: np.ndarray,
    sr: float = 400.0,
    onset: float = 0.3,
    offset: float = 0.15,
    min_duration_s: float = 3.0,
    min_gap_s: float = 2.0,
    collar_s: float = 0.0,
    chunk_size: int = 1_000_000,
) -> list[tuple[int, int]]:
    """Return hysteresis segments as sample-index intervals [start, end)."""
    probs = np.asarray(probs).reshape(-1)
    raw_segments = hysteresis_raw_segments(
        probs,
        onset=onset,
        offset=offset,
        chunk_size=chunk_size,
    )
    return finalize_hysteresis_segments(
        raw_segments,
        n_samples=int(probs.size),
        sr=sr,
        min_duration_s=min_duration_s,
        min_gap_s=min_gap_s,
        collar_s=collar_s,
    )


def postprocess(
    probs: np.ndarray,
    sr: float = 400.0,
    onset: float = 0.3,
    offset: float = 0.15,
    min_duration_s: float = 3.0,
    min_gap_s: float = 2.0,
    collar_s: float = 0.0,
) -> list[tuple[float, float]]:
    segments = postprocess_samples(
        probs=probs,
        sr=sr,
        onset=onset,
        offset=offset,
        min_duration_s=min_duration_s,
        min_gap_s=min_gap_s,
        collar_s=collar_s,
    )
    return [(start / sr, end / sr) for start, end in segments]


def interval_positive_samples(segments) -> int:
    arr = _as_segments_array(segments)
    if arr.size == 0:
        return 0
    lengths = arr[:, 1] - arr[:, 0]
    lengths = lengths[lengths > 0]
    if lengths.size == 0:
        return 0
    return int(lengths.sum())


def interval_overlap_samples(predicted, target) -> int:
    predicted = _merge_segments_array(predicted, assume_sorted=True)
    target = _merge_segments_array(target, assume_sorted=True)
    if predicted.size == 0 or target.size == 0:
        return 0

    pred_starts = predicted[:, 0]
    pred_ends = predicted[:, 1]
    total = 0

    for target_start, target_end in target:
        left = int(np.searchsorted(pred_ends, target_start, side="right"))
        right = int(np.searchsorted(pred_starts, target_end, side="left"))
        if right <= left:
            continue
        overlap_starts = np.maximum(pred_starts[left:right], target_start)
        overlap_ends = np.minimum(pred_ends[left:right], target_end)
        overlaps = overlap_ends - overlap_starts
        total += int(overlaps[overlaps > 0].sum())

    return int(total)


def event_match_count(predicted, target, iou_threshold: float = 0.0, min_overlap_samples: int = 1) -> int:
    predicted = _merge_segments_array(predicted, assume_sorted=True)
    target = _merge_segments_array(target, assume_sorted=True)
    if predicted.size == 0 or target.size == 0:
        return 0

    pred_starts = predicted[:, 0]
    pred_ends = predicted[:, 1]
    used_pred = np.zeros(predicted.shape[0], dtype=bool)
    matches = 0

    for target_start, target_end in target:
        left = int(np.searchsorted(pred_ends, target_start, side="right"))
        right = int(np.searchsorted(pred_starts, target_end, side="left"))
        if right <= left:
            continue

        candidate_idx = np.arange(left, right)
        unused = ~used_pred[candidate_idx]
        if not bool(np.any(unused)):
            continue
        candidate_idx = candidate_idx[unused]

        overlap_starts = np.maximum(pred_starts[candidate_idx], target_start)
        overlap_ends = np.minimum(pred_ends[candidate_idx], target_end)
        overlaps = overlap_ends - overlap_starts
        valid = overlaps >= min_overlap_samples
        if not bool(np.any(valid)):
            continue

        candidate_idx = candidate_idx[valid]
        overlaps = overlaps[valid]
        union_starts = np.minimum(pred_starts[candidate_idx], target_start)
        union_ends = np.maximum(pred_ends[candidate_idx], target_end)
        unions = union_ends - union_starts
        ious = np.divide(
            overlaps,
            unions,
            out=np.zeros(overlaps.shape, dtype=np.float64),
            where=unions > 0,
        )
        valid_iou = ious >= iou_threshold
        if not bool(np.any(valid_iou)):
            continue

        candidate_idx = candidate_idx[valid_iou]
        ious = ious[valid_iou]
        best_idx = int(candidate_idx[int(np.argmax(ious))])
        if not bool(used_pred[best_idx]):
            used_pred[best_idx] = True
            matches += 1

    return int(matches)


def precision_recall_f1(tp: int, pred: int, true: int) -> tuple[float, float, float]:
    precision = tp / pred if pred else 0.0
    recall = tp / true if true else 0.0
    f1 = 2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    return precision, recall, f1


def evaluate_segments(
    predicted,
    target,
    event_iou_threshold: float = 0.0,
    min_event_overlap_samples: int = 1,
) -> SegmentMetrics:
    predicted = _merge_segments_array(predicted, assume_sorted=True)
    target = _merge_segments_array(target, assume_sorted=True)

    sample_tp = interval_overlap_samples(predicted, target)
    sample_pred = interval_positive_samples(predicted)
    sample_true = interval_positive_samples(target)
    sample_precision, sample_recall, sample_f1 = precision_recall_f1(
        sample_tp, sample_pred, sample_true
    )

    event_tp = event_match_count(
        predicted,
        target,
        iou_threshold=event_iou_threshold,
        min_overlap_samples=min_event_overlap_samples,
    )
    event_pred = int(predicted.shape[0])
    event_true = int(target.shape[0])
    event_precision, event_recall, event_f1 = precision_recall_f1(
        event_tp, event_pred, event_true
    )

    return SegmentMetrics(
        sample_precision=sample_precision,
        sample_recall=sample_recall,
        sample_f1=sample_f1,
        event_precision=event_precision,
        event_recall=event_recall,
        event_f1=event_f1,
        sample_tp=sample_tp,
        sample_pred=sample_pred,
        sample_true=sample_true,
        event_tp=event_tp,
        event_pred=event_pred,
        event_true=event_true,
    )
