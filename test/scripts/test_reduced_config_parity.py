#!/usr/bin/env python3
"""Regression tests for reduced/full configuration parity policy."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from check_reduced_config_parity import ConfigPair, check_pair


class ReducedConfigParityTest(unittest.TestCase):
    def check_texts(
        self,
        full_text: str,
        reduced_text: str,
        allowed_differences: frozenset[str] = frozenset(),
    ) -> tuple[list[str], int, int]:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            full = root / "full.config"
            reduced = root / "reduced.config"
            full.write_text(full_text, encoding="utf-8")
            reduced.write_text(reduced_text, encoding="utf-8")
            return check_pair(
                ConfigPair(
                    label="test",
                    full=full,
                    reduced=reduced,
                    allowed_differences=allowed_differences,
                )
            )

    def test_equal_new_field_is_checked_automatically(self) -> None:
        errors, field_count, allowed_count = self.check_texts(
            "-existing 1\n-new_field 2\n",
            "-existing 1\n-new_field 2\n",
        )
        self.assertEqual(errors, [])
        self.assertEqual(field_count, 2)
        self.assertEqual(allowed_count, 0)

    def test_missing_new_field_is_rejected_by_default(self) -> None:
        errors, field_count, allowed_count = self.check_texts(
            "-existing 1\n-new_field 2\n",
            "-existing 1\n",
        )
        self.assertEqual(field_count, 2)
        self.assertEqual(allowed_count, 0)
        self.assertEqual(
            errors,
            ["-new_field: reduced='<missing>', full='2'"],
        )

    def test_only_explicitly_allowed_difference_is_accepted(self) -> None:
        errors, field_count, allowed_count = self.check_texts(
            "-gpgpu_n_clusters 12\n-shared 1\n",
            "-gpgpu_n_clusters 1\n-shared 1\n",
            frozenset({"-gpgpu_n_clusters"}),
        )
        self.assertEqual(errors, [])
        self.assertEqual(field_count, 2)
        self.assertEqual(allowed_count, 1)


if __name__ == "__main__":
    unittest.main()
