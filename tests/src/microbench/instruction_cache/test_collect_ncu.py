#!/usr/bin/env python3

import importlib.util
import unittest
from pathlib import Path


MODULE_PATH = Path(__file__).with_name("collect_ncu.py")
SPEC = importlib.util.spec_from_file_location("collect_ncu", MODULE_PATH)
collect_ncu = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(collect_ncu)


class CollectNcuTest(unittest.TestCase):
    def test_parse_ncu_csv_ignores_profiler_messages_and_units(self):
        output = """==PROF== Connected\nprogram output\n\"ID\",\"Kernel Name\",\"sm__icc_requests.sum\"\n\"\",\"\",\"cycle\"\n\"0\",\"footprint_kernel()\",\"42\"\n"""
        row = collect_ncu.parse_ncu_csv(output)
        self.assertEqual(row["Kernel Name"], "footprint_kernel()")
        self.assertEqual(row["sm__icc_requests.sum"], "42")

    def test_parse_ncu_csv_rejects_multiple_profiled_launches(self):
        output = """\"ID\",\"Kernel Name\"\n\"\",\"\"\n\"0\",\"a\"\n\"1\",\"b\"\n"""
        with self.assertRaisesRegex(ValueError, "expected one"):
            collect_ncu.parse_ncu_csv(output)

    def test_derived_values_check_counter_identities(self):
        row = {
            "sm__icc_requests.sum": "100",
            "sm__icc_requests_lookup_hit.sum": "70",
            "sm__icc_requests_lookup_miss.sum": "30",
            "gcc__cache_requests_type_instruction.sum": "20",
            "gcc__cache_requests_type_instruction_lookup_hit.sum": "12",
            "gcc__cache_requests_type_instruction_lookup_miss.sum": "8",
            "gcc__gcc2xbar_requests_type_instruction.sum": "4",
            "gcc__cycles_elapsed.sum": "1100",
            "gcc__cycles_elapsed.avg": "100",
            "lts__t_sectors_srcunit_gcc.sum": "32",
            "smsp__inst_executed.sum": "160",
        }
        derived = collect_ncu.derived_values(row)
        self.assertEqual(derived["icc_miss_rate"], "0.3")
        self.assertEqual(derived["gcc_miss_rate"], "0.4")
        self.assertEqual(derived["inst_per_gcc_request"], "8")
        self.assertEqual(derived["lts_sectors_per_gcc2xbar_request"], "8")
        self.assertEqual(derived["gcc_instances"], "11")
        self.assertEqual(derived["icc_identity_error"], "0")
        self.assertEqual(derived["gcc_identity_error"], "0")


if __name__ == "__main__":
    unittest.main()
