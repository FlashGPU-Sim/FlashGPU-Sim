#!/usr/bin/env python3

import io
import unittest

from analyze_cta_lifecycle import LifecycleError, read_lifecycles, summarize


def event(text: str) -> str:
    return f"GPGPU-Sim Cycle 1: LIVENESS - Core 0 - CTA_LIFECYCLE {text}\n"


class CtaLifecycleAnalysisTest(unittest.TestCase):
    def test_valid_lifecycles_are_summarized(self) -> None:
        log = io.StringIO(
            event(
                "event=admit kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=10 slot_reuse=0 "
                "previous_release=0 release_to_admit=0"
            )
            + event(
                "event=threads_exit kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=20 pending_tma=0"
            )
            + event(
                "event=release kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=20 active_cycles=10 threads_exit=20 "
                "exit_to_release=0 pending_tma=0 replacement_ready=21 "
                "replacement_latency=1"
            )
            + event(
                "event=admit kernel_uid=7 sid=0 hw_cta=0 logical_cta=2 "
                "generation=2 cycle=21 slot_reuse=1 "
                "previous_release=20 release_to_admit=1"
            )
            + event(
                "event=threads_exit kernel_uid=7 sid=0 hw_cta=0 logical_cta=2 "
                "generation=2 cycle=30 pending_tma=1"
            )
            + event(
                "event=release kernel_uid=7 sid=0 hw_cta=0 logical_cta=2 "
                "generation=2 cycle=34 active_cycles=13 threads_exit=30 "
                "exit_to_release=4 pending_tma=1 replacement_ready=35 "
                "replacement_latency=1"
            )
            + event(
                "event=admit kernel_uid=7 sid=1 hw_cta=0 logical_cta=1 "
                "generation=1 cycle=10 slot_reuse=0 "
                "previous_release=0 release_to_admit=0"
            )
            + event(
                "event=threads_exit kernel_uid=7 sid=1 hw_cta=0 logical_cta=1 "
                "generation=1 cycle=25 pending_tma=0"
            )
            + event(
                "event=release kernel_uid=7 sid=1 hw_cta=0 logical_cta=1 "
                "generation=1 cycle=25 active_cycles=15 threads_exit=25 "
                "exit_to_release=0 pending_tma=0 replacement_ready=26 "
                "replacement_latency=1"
            )
        )
        summary = summarize(read_lifecycles(log))
        kernel = summary["kernels"]["7"]
        self.assertEqual(kernel["ctas"], 3)
        self.assertEqual(kernel["sms"], 2)
        self.assertEqual(kernel["slot_initial_admits"], 2)
        self.assertEqual(kernel["slot_replacement_admits"], 1)
        self.assertEqual(kernel["duplicate_logical_cta_ids"], 0)
        self.assertEqual(kernel["ctas_per_sm"]["min"], 1)
        self.assertEqual(kernel["ctas_per_sm"]["max"], 2)
        self.assertEqual(kernel["release_to_admit"]["mean"], 1)
        self.assertEqual(kernel["pending_tma_drain"]["mean"], 4)
        self.assertEqual(kernel["sm_active_span"]["max"], 24)
        self.assertEqual(kernel["sm_last_release_spread"], 9)

    def test_release_without_exit_is_rejected(self) -> None:
        log = io.StringIO(
            event(
                "event=admit kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=10 slot_reuse=0 "
                "previous_release=0 release_to_admit=0"
            )
            + event(
                "event=release kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=20 active_cycles=10 threads_exit=20 "
                "exit_to_release=0 pending_tma=0"
            )
        )
        with self.assertRaisesRegex(LifecycleError, "precedes threads_exit"):
            read_lifecycles(log)

    def test_replacement_before_ready_is_rejected(self) -> None:
        log = io.StringIO(
            event(
                "event=admit kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=10 slot_reuse=0 "
                "previous_release=0 release_to_admit=0"
            )
            + event(
                "event=threads_exit kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=20 pending_tma=0"
            )
            + event(
                "event=release kernel_uid=7 sid=0 hw_cta=0 logical_cta=0 "
                "generation=1 cycle=20 active_cycles=10 threads_exit=20 "
                "exit_to_release=0 pending_tma=0 replacement_ready=25 "
                "replacement_latency=5"
            )
            + event(
                "event=admit kernel_uid=7 sid=0 hw_cta=0 logical_cta=1 "
                "generation=2 cycle=24 slot_reuse=1 "
                "previous_release=20 release_to_admit=4"
            )
            + event(
                "event=threads_exit kernel_uid=7 sid=0 hw_cta=0 logical_cta=1 "
                "generation=2 cycle=30 pending_tma=0"
            )
            + event(
                "event=release kernel_uid=7 sid=0 hw_cta=0 logical_cta=1 "
                "generation=2 cycle=30 active_cycles=6 threads_exit=30 "
                "exit_to_release=0 pending_tma=0 replacement_ready=35 "
                "replacement_latency=5"
            )
        )
        with self.assertRaisesRegex(LifecycleError, "replacement admitted before ready"):
            read_lifecycles(log)


if __name__ == "__main__":
    unittest.main()
