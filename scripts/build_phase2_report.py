"""Assemble the canonical Phase 2 sim2real closeout report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml


def _load_json(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_phase2_report(
    flat_summary: str | Path,
    slope_summary: str | Path,
    stairs_summary: str | Path,
    anymal_config: str | Path,
    anymal_alignment: str | Path,
    output_path: str | Path = "results/sim2real_report.json",
) -> Path:
    report = {
        "phase": "phase2",
        "go1": {
            "flat": _load_json(flat_summary),
            "slope": _load_json(slope_summary),
            "stairs": _load_json(stairs_summary),
            "notes": [
                "Slope and stairs canonical outputs use selected-model warm-start curriculum logic.",
                "Flat uses the best checkpoint from the dedicated flat run.",
            ],
        },
        "anymal_d": {
            "config": _load_yaml(anymal_config),
            "config_yaml_path": str(anymal_config),
            "alignment": _load_json(anymal_alignment),
        },
    }
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with open(output, "w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return output


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the final Phase 2 sim2real report")
    parser.add_argument("--flat-summary", default="checkpoints/phase2/go1/flat/summary.json")
    parser.add_argument("--slope-summary", default="checkpoints/phase2/go1/slope/summary.json")
    parser.add_argument("--stairs-summary", default="checkpoints/phase2/go1/stairs/summary.json")
    parser.add_argument("--anymal-config", default="configs/sim2real/anymal_d.yaml")
    parser.add_argument("--anymal-alignment", default="configs/sim2real/anymal_d.alignment.json")
    parser.add_argument("--output", default="results/sim2real_report.json")
    args = parser.parse_args()

    build_phase2_report(
        flat_summary=args.flat_summary,
        slope_summary=args.slope_summary,
        stairs_summary=args.stairs_summary,
        anymal_config=args.anymal_config,
        anymal_alignment=args.anymal_alignment,
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
