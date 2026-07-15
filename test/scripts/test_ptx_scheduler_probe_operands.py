#!/usr/bin/env python3
"""Regression tests for PTX scheduler probe def/use classification."""

from __future__ import annotations

import unittest

import ptx_sass_guided_scheduler_probe as sass_guided
import ptx_window_scheduler_probe as window


PROBES = (sass_guided, window)


class OperandClassificationTest(unittest.TestCase):
    def assert_classification(
        self,
        op: str,
        operands: str,
        pred: str | None,
        expected_defs: set[str],
        expected_uses: set[str],
    ) -> None:
        for probe in PROBES:
            with self.subTest(probe=probe.__name__, op=op):
                defs, uses = probe.extract_ptx_defs_uses(op, operands, pred)
                self.assertEqual(defs, expected_defs)
                self.assertEqual(uses, expected_uses)

    def test_setp_has_one_destination_and_source_operands(self) -> None:
        self.assert_classification(
            "setp.ne.u32",
            "%p1, %r1, 0",
            None,
            {"%p1"},
            {"%r1"},
        )

    def test_setp_dual_destination_and_predicate_input(self) -> None:
        self.assert_classification(
            "setp.lt.and.s32",
            "%p1|%p2, %r1, %r2, %p3",
            "%p_guard",
            {"%p1", "%p2"},
            {"%r1", "%r2", "%p3", "%p_guard"},
        )

    def test_regular_instruction_still_defines_first_operand(self) -> None:
        self.assert_classification(
            "add.u32",
            "%r3, %r1, %r2",
            None,
            {"%r3"},
            {"%r1", "%r2"},
        )


if __name__ == "__main__":
    unittest.main()
