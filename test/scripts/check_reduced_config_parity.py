#!/usr/bin/env python3
"""Check full/reduced config parity for supported GPU configurations."""

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, FrozenSet, List, Tuple


REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class ConfigPair:
    label: str
    full: Path
    reduced: Path
    allowed_differences: FrozenSet[str]


# The policy is intentionally expressed only as fields that reduced configs may
# change or omit. Every other field found in either file must exist in both and
# have the same value. New fields therefore inherit parity checking
# automatically.
CONFIG_PAIRS = {
    "h100": ConfigPair(
        label="SM90 H100",
        full=REPO_ROOT / "configs" / "SM90_H100" / "gpgpusim.config",
        reduced=(
            REPO_ROOT
            / "configs"
            / "SM90_H100_REDUCED"
            / "gpgpusim.config"
        ),
        allowed_differences=frozenset(
            {
                "-gpgpu_n_clusters",
                "-gpgpu_n_mem",
                "-gpgpu_n_sub_partition_per_mchannel",
                "-gpgpu_mem_addr_mapping",
                "-gpgpu_cache:dl2",
                "-gpgpu_l2_partition_count",
                "-gpgpu_l2_partition_extra_latency",
                "-gpgpu_ipoly_non_power2_balanced",
                "-gpgpu_ipoly_channel_stable_l2slice",
                "-gpgpu_clock_domains",
                "-gpgpu_dram_buswidth",
                "-gpgpu_dram_burst_length",
                "-dram_data_command_freq_ratio",
                "-dram_dual_bus_interface",
                "-gpgpu_dram_timing_opt",
                "-gpgpu_shmem_sizeDefault",
                "-gpgpu_max_dynamic_smem_prefer_occupancy_carveout",
                "-gpgpu_tma_request_granularity",
                "-trace_sampling_core",
            }
        ),
    ),
    "rtx5090": ConfigPair(
        label="SM120 RTX5090",
        full=REPO_ROOT / "configs" / "SM120_RTX5090" / "gpgpusim.config",
        reduced=(
            REPO_ROOT
            / "configs"
            / "SM120_RTX5090_REDUCED"
            / "gpgpusim.config"
        ),
        allowed_differences=frozenset(
            {
                "-gpgpu_n_clusters",
            }
        ),
    ),
}


def parse_config(path: Path) -> Tuple[Dict[str, str], List[str]]:
    options: Dict[str, str] = {}
    locations: Dict[str, int] = {}
    errors: List[str] = []

    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except OSError as exc:
        return {}, [f"cannot read {path}: {exc}"]

    for lineno, raw_line in enumerate(lines, start=1):
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        fields = line.split(None, 1)
        key = fields[0]
        if not key.startswith("-"):
            errors.append(f"{path}:{lineno}: unexpected config line: {raw_line}")
            continue
        value = fields[1].strip() if len(fields) == 2 else ""
        if key in options:
            errors.append(
                f"{path}:{lineno}: duplicate {key}; first defined at "
                f"line {locations[key]}"
            )
            continue
        options[key] = value
        locations[key] = lineno

    return options, errors


def display_value(options: Dict[str, str], key: str) -> str:
    return options.get(key, "<missing>")


def check_pair(pair: ConfigPair) -> Tuple[List[str], int, int]:
    full, errors = parse_config(pair.full)
    reduced, reduced_errors = parse_config(pair.reduced)
    errors.extend(reduced_errors)
    if errors:
        return errors, 0, 0

    all_fields = set(full) | set(reduced)
    actual_differences = {
        key for key in all_fields if full.get(key) != reduced.get(key)
    }
    disallowed_differences = actual_differences - pair.allowed_differences

    for key in sorted(disallowed_differences):
        errors.append(
            f"{key}: reduced={display_value(reduced, key)!r}, "
            f"full={display_value(full, key)!r}"
        )

    allowed_count = len(actual_differences & pair.allowed_differences)
    return errors, len(all_fields), allowed_count


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--pair",
        action="append",
        choices=sorted(CONFIG_PAIRS),
        help="configuration pair to check; may be repeated (default: all)",
    )
    args = parser.parse_args()

    selected_names = args.pair or list(CONFIG_PAIRS)
    failed = False
    for name in selected_names:
        pair = CONFIG_PAIRS[name]
        errors, field_count, allowed_count = check_pair(pair)
        if errors:
            failed = True
            print(f"{pair.label} reduced/full config parity check failed:")
            for error in errors:
                print(f"  - {error}")
            continue

        print(
            f"{pair.label} reduced/full config parity check passed: "
            f"{field_count} fields checked; "
            f"{allowed_count} allowed differences present."
        )

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
