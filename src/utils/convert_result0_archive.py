#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Convert the result_0 MESD annotations with matching EDF_july EDF files into
the training-ready processed layout.

Output layout:
data/processed2/
  <animal_id>/
    <session_id>/
      processed_signals.npy
      seizure_mask.npy
      conversion_metadata.json
      segments_info.csv
  conversion_summary.csv
  result0_pairing_report.csv
"""

import argparse
import csv
import json
import site
import sys
from pathlib import Path
from typing import Dict, List


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
site.addsitedir(str(PROJECT_ROOT / ".codex_pydeps"))

from src.utils.edf_converter import EDFConverter  # noqa: E402


def annotation_key(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_MESD"):
        stem = stem[:-5]
    return stem


def split_recording_id(recording_id: str) -> tuple[str, str]:
    parts = recording_id.split("_", 1)
    if len(parts) == 1:
        return recording_id, "session"
    return parts[0], parts[1]


def build_pairs(edf_dir: Path, annotations_dir: Path) -> tuple[List[Dict], List[Dict]]:
    edf_by_key = {path.stem: path for path in sorted(edf_dir.glob("*.edf"))}
    annotations_by_key = {
        annotation_key(path): path
        for path in sorted(annotations_dir.glob("*_MESD.txt"))
    }

    pairs = []
    issues = []
    for key, annotation_path in annotations_by_key.items():
        edf_path = edf_by_key.get(key)
        if edf_path is None:
            issues.append({"recording_id": key, "issue": "missing_edf", "path": str(annotation_path)})
            continue

        animal_id, session_id = split_recording_id(key)
        pairs.append(
            {
                "recording_id": key,
                "animal_id": animal_id,
                "session_id": session_id,
                "edf_file": edf_path,
                "annotation_file": annotation_path,
            }
        )

    for key, edf_path in edf_by_key.items():
        if key not in annotations_by_key:
            issues.append({"recording_id": key, "issue": "missing_annotation", "path": str(edf_path)})

    return pairs, issues


def write_pairing_report(output_dir: Path, pairs: List[Dict], issues: List[Dict]) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    report_path = output_dir / "result0_pairing_report.csv"
    fieldnames = ["recording_id", "animal_id", "session_id", "edf_file", "annotation_file", "issue"]

    with report_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for pair in pairs:
            writer.writerow(
                {
                    "recording_id": pair["recording_id"],
                    "animal_id": pair["animal_id"],
                    "session_id": pair["session_id"],
                    "edf_file": pair["edf_file"],
                    "annotation_file": pair["annotation_file"],
                    "issue": "",
                }
            )
        for issue in issues:
            writer.writerow(
                {
                    "recording_id": issue["recording_id"],
                    "animal_id": "",
                    "session_id": "",
                    "edf_file": issue["path"] if issue["issue"] == "missing_annotation" else "",
                    "annotation_file": issue["path"] if issue["issue"] == "missing_edf" else "",
                    "issue": issue["issue"],
                }
            )


def is_complete(output_dir: Path, pair: Dict) -> bool:
    session_dir = output_dir / pair["animal_id"] / pair["session_id"]
    required_files = [
        "processed_signals.npy",
        "seizure_mask.npy",
        "segments_info.csv",
        "conversion_metadata.json",
    ]
    return all((session_dir / name).exists() for name in required_files)


def metadata_to_summary_row(metadata: Dict, status: str = "success", error: str = "") -> Dict:
    return {
        "animal_id": metadata["animal_id"],
        "session_id": metadata["session_id"],
        "status": status,
        "n_seizures": metadata.get("n_seizures", ""),
        "seizure_duration": metadata.get("seizure_duration", ""),
        "duration": metadata.get("duration", ""),
        "sampling_freq": metadata.get("sampling_freq", ""),
        "error": error,
    }


def write_summary(output_dir: Path, rows: List[Dict]) -> None:
    summary_path = output_dir / "conversion_summary.csv"
    fieldnames = [
        "animal_id",
        "session_id",
        "status",
        "n_seizures",
        "seizure_duration",
        "duration",
        "sampling_freq",
        "error",
    ]
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def summarize_existing(output_dir: Path) -> List[Dict]:
    rows = []
    for metadata_path in sorted(output_dir.glob("*/*/conversion_metadata.json")):
        with metadata_path.open("r", encoding="utf-8") as f:
            metadata = json.load(f)
        rows.append(metadata_to_summary_row(metadata))
    write_summary(output_dir, rows)
    return rows


def convert_pairs(pairs: List[Dict], output_dir: Path, target_sampling_rate: float, limit: int | None) -> Dict:
    converter = EDFConverter(target_sampling_rate=target_sampling_rate)
    selected_pairs = pairs[:limit] if limit is not None else pairs
    results = {
        "total_files": len(selected_pairs),
        "successful_conversions": 0,
        "failed_conversions": 0,
        "details": [],
    }

    for pair in selected_pairs:
        print(f"\nProcessing: {pair['animal_id']}/{pair['session_id']}")
        conversion_result = converter.convert_single_file(
            str(pair["edf_file"]),
            str(pair["annotation_file"]),
            str(output_dir),
            pair["animal_id"],
            pair["session_id"],
        )
        results["details"].append({**pair, "result": conversion_result})
        converter.edf_loader.close_all()
        if conversion_result["success"]:
            results["successful_conversions"] += 1
        else:
            results["failed_conversions"] += 1

    rows = []
    for detail in results["details"]:
        result = detail["result"]
        if result["success"]:
            rows.append(metadata_to_summary_row(result["metadata"]))
        else:
            rows.append(
                {
                    "animal_id": detail["animal_id"],
                    "session_id": detail["session_id"],
                    "status": "failed",
                    "n_seizures": "",
                    "seizure_duration": "",
                    "duration": "",
                    "sampling_freq": "",
                    "error": result.get("error", "Unknown error"),
                }
            )
    write_summary(output_dir, rows)

    return results


def main() -> None:
    parser = argparse.ArgumentParser(description="Convert result_0 + EDF_july into data/processed2")
    parser.add_argument("--edf_dir", default="data/orig/EDF_july")
    parser.add_argument("--annotations_dir", default="data/orig/result_0/MESD")
    parser.add_argument("--output_dir", default="data/processed2")
    parser.add_argument("--target_sampling_rate", type=float, default=400.0)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--recording_id", default=None)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--summarize-existing", action="store_true")
    args = parser.parse_args()

    edf_dir = Path(args.edf_dir)
    annotations_dir = Path(args.annotations_dir)
    output_dir = Path(args.output_dir)

    pairs, issues = build_pairs(edf_dir, annotations_dir)
    if args.recording_id:
        pairs = [pair for pair in pairs if pair["recording_id"] == args.recording_id]
        if not pairs:
            raise ValueError(f"Recording id not found: {args.recording_id}")
    if args.skip_existing:
        pairs = [pair for pair in pairs if not is_complete(output_dir, pair)]
    write_pairing_report(output_dir, pairs, issues)

    print(f"Matched pairs: {len(pairs)}")
    print(f"Pairing issues: {len(issues)}")
    if issues:
        for issue in issues:
            print(f"{issue['issue']}: {issue['recording_id']}")

    if args.dry_run:
        return

    if args.summarize_existing:
        rows = summarize_existing(output_dir)
        print(f"Existing converted sessions: {len(rows)}")
        return

    results = convert_pairs(pairs, output_dir, args.target_sampling_rate, args.limit)
    print("\nConversion results:")
    print(f"Total files: {results['total_files']}")
    print(f"Successful: {results['successful_conversions']}")
    print(f"Failed: {results['failed_conversions']}")


if __name__ == "__main__":
    main()
