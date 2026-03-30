"""CLI wrapper for the OLSD license gating utility."""

from __future__ import annotations

import sys

from olsd.pipeline.license_check import check_hf_license


def _safe_print(message: str) -> None:
    cleaned = message.encode(sys.stdout.encoding or "utf-8", errors="replace").decode(
        sys.stdout.encoding or "utf-8",
        errors="replace",
    )
    print(cleaned)


def main() -> int:
    repo_ids = sys.argv[1:] or [
        "leggedrobotics/grand_tour_dataset",
        "openhe/g1-retargeted-motions",
    ]
    blocked = False
    for repo_id in repo_ids:
        result = check_hf_license(repo_id)
        _safe_print(result["message"])
        blocked = blocked or result["blocked"]
    return 1 if blocked else 0


if __name__ == "__main__":
    raise SystemExit(main())
