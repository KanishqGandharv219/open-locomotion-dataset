"""
License Gating Utility — Check dataset licenses before ingestion.

Queries the HuggingFace dataset card's license field and aborts or flags
non-permissive licenses. Keeps the redistribution policy transparent.
"""

from __future__ import annotations

import logging
import sys

logger = logging.getLogger(__name__)

# Licenses that are safe to redistribute in OLSD (permissive)
PERMISSIVE_LICENSES = {
    "cc-by-4.0", "cc-by-3.0", "cc-by-2.0",
    "cc0-1.0", "cc0",
    "mit", "apache-2.0", "bsd-2-clause", "bsd-3-clause",
    "unlicense", "odc-by", "odc-odbl", "pddl",
    "openrail", "openrail++",
}

# Licenses that allow benchmarking but NOT redistribution
BENCHMARK_ONLY_LICENSES = {
    "cc-by-nc-4.0", "cc-by-nc-3.0",
    "cc-by-nc-sa-4.0", "cc-by-nc-sa-3.0",
    "cc-by-nd-4.0", "cc-by-sa-4.0",
    "gpl-3.0", "lgpl-3.0", "agpl-3.0",
}


def check_hf_license(repo_id: str) -> dict:
    """
    Check the license of a HuggingFace dataset.

    Args:
        repo_id: HuggingFace dataset ID (e.g., "leggedrobotics/grand_tour_dataset").

    Returns:
        Dict with keys:
            license: str            - license identifier
            permissive: bool        - safe for redistribution
            benchmark_only: bool    - can use for benchmarking, not redistribution
            blocked: bool           - unknown/restrictive, manual review needed
            message: str            - human-readable status message
    """
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.dataset_info(repo_id)
        license_id = (info.card_data.get("license", "unknown") if info.card_data else "unknown")
    except Exception as e:
        logger.warning(f"Could not fetch license for {repo_id}: {e}")
        license_id = "unknown"

    license_lower = license_id.lower().strip()

    if license_lower in PERMISSIVE_LICENSES:
        return {
            "license": license_id,
            "permissive": True,
            "benchmark_only": False,
            "blocked": False,
            "message": f"✅ {repo_id}: {license_id} — permissive, safe for redistribution",
        }
    elif license_lower in BENCHMARK_ONLY_LICENSES:
        return {
            "license": license_id,
            "permissive": False,
            "benchmark_only": True,
            "blocked": False,
            "message": f"⚠️ {repo_id}: {license_id} — benchmark-only, do NOT redistribute",
        }
    else:
        return {
            "license": license_id,
            "permissive": False,
            "benchmark_only": False,
            "blocked": True,
            "message": f"🚫 {repo_id}: {license_id} — unknown license, manual review required",
        }


def gate_ingestion(repo_id: str, allow_benchmark: bool = False) -> bool:
    """
    Gate ingestion based on license check.

    Args:
        repo_id: HuggingFace dataset ID.
        allow_benchmark: If True, allow benchmark-only licenses.

    Returns:
        True if ingestion is allowed, False otherwise.
    """
    result = check_hf_license(repo_id)
    logger.info(result["message"])

    if result["permissive"]:
        return True
    elif result["benchmark_only"] and allow_benchmark:
        logger.warning("Proceeding with benchmark-only license. Data will NOT be redistributed.")
        return True
    else:
        logger.error(f"Ingestion blocked for {repo_id}. License: {result['license']}")
        return False


if __name__ == "__main__":
    """CLI usage: python -m olsd.pipeline.license_check <repo_id>"""
    logging.basicConfig(level=logging.INFO)

    datasets_to_check = [
        "leggedrobotics/grand_tour_dataset",
        "openhe/g1-retargeted-motions",
    ]

    if len(sys.argv) > 1:
        datasets_to_check = sys.argv[1:]

    for repo_id in datasets_to_check:
        result = check_hf_license(repo_id)
        print(result["message"])
