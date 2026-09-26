#!/usr/bin/env python3

from __future__ import annotations

import unittest
import tempfile
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts"))

from generate_sass_ptxline_guide import add_line_markers, instruction_stream, normalize_ptx, generate_guide


class AddLineMarkersTest(unittest.TestCase):
    def test_marks_only_instructions_and_preserves_original_line_numbers(self) -> None:
        source = (
            ".version 9.1\n"
            ".target sm_100a\n"
            ".address_size 64\n"
            ".visible .entry first()\n"
            "{\n"
            "  .reg .b32 %r<3>;\n"
            "$L0:\n"
            "  mov.b32 %r1, %r2;\n"
            "  @%p0 bra $L0;\n"
            "}\n"
        )

        marked, count = add_line_markers(source, 'input"name.ptx')

        self.assertEqual(count, 2)
        self.assertIn('.file 1 "input\\"name.ptx"\n', marked)
        self.assertIn(".loc 1 8 0\n  mov.b32 %r1, %r2;", marked)
        self.assertIn(".loc 1 9 0\n  @%p0 bra $L0;", marked)
        self.assertNotIn(".loc 1 6 0", marked)

    def test_uses_an_unused_file_index_for_existing_debug_files(self) -> None:
        source = (
            ".address_size 64\n"
            '.file 3 "upstream.cu"\n'
            ".visible .entry kernel() {\n"
            "  ret;\n"
            "}\n"
        )

        marked, count = add_line_markers(source, "kernel.ptx")

        self.assertEqual(count, 1)
        self.assertIn('.file 4 "kernel.ptx"', marked)
        self.assertIn(".loc 4 4 0\n  ret;", marked)

    def test_marks_device_function_but_not_external_declaration(self) -> None:
        source = (
            ".address_size 64\n.extern .func external();\n"
            ".func helper() {\n  ret;\n}\n"
            ".visible .entry kernel() {\n  ret;\n}\n"
        )
        marked, count = add_line_markers(source, "module.ptx")
        self.assertEqual(count, 2)
        self.assertIn(".loc 1 4 0\n  ret;", marked)
        self.assertIn(".loc 1 7 0\n  ret;", marked)

    def test_comment_braces_do_not_end_function_or_hide_instructions(self) -> None:
        source = (
            ".address_size 64\n"
            '.file 3 "path//with/*braces}*/.ptx"\n'
            "/* .entry fake() { ; */\n"
            ".visible .entry kernel() {\n"
            "  // } not a function end\n"
            "  /* }\n"
            "  .func fake(); { */ mov.u32 %r1, %r2; // }\n"
            "  ret; /* { */\n"
            "}\n"
        )
        marked, count = add_line_markers(source, "kernel.ptx")
        self.assertEqual(count, 2)
        self.assertIn('.file 3 "path//with/*braces}*/.ptx"', marked)
        self.assertIn('.file 4 "kernel.ptx"', marked)
        # A marker before a block-comment closing line must remain outside
        # the comment, so ptxas can see it.
        self.assertRegex(marked, r"\.loc 4 7 0\n[ ]*mov.u32 %r1, %r2;")
        self.assertRegex(marked, r"\.loc 4 8 0\n[ ]*ret;")
        self.assertEqual(marked.count(".loc "), 2)

    def test_multiline_instruction_marker_precedes_the_whole_instruction(self) -> None:
        source = ".address_size 64\n.entry kernel() {\nadd.u32\n %a, %b, 1;\nret;\n}\n"
        marked, count = add_line_markers(source, "input.ptx")
        self.assertEqual(count, 2)
        self.assertIn(".loc 1 4 0\nadd.u32\n %a, %b, 1;", marked)
        normalized = normalize_ptx(source)
        self.assertIn("\nadd.u32 %a, %b, 1;\n", normalized)

    def test_compact_body_labels_and_operand_braces(self) -> None:
        source = (".address_size 64\n.entry kernel() { .reg .b32 %r<3>; "
                  "{ $L: mov.b64 {%r1, %r2}, %rd; } ret; }\n")
        normalized = normalize_ptx(source)
        self.assertIn("\nmov.b64 {%r1, %r2}, %rd;\n", normalized)
        self.assertIn("\nret;\n", normalized)
        marked, count = add_line_markers(normalized, "input.ptx")
        self.assertEqual(count, 2)
        for number, line in enumerate(normalized.splitlines(), 1):
            if line.startswith(("mov.", "ret;")):
                self.assertIn(f".loc 1 {number} 0\n{line}", marked)

    def test_opcode_scope_qualifiers_are_not_labels(self) -> None:
        source = (".address_size 64\n.entry kernel() { "
                  "mbarrier.arrive.release.cta.shared::cta.b64 %s, [%a]; "
                  "tcgen05.wait::st.sync.aligned; }\n")
        normalized = normalize_ptx(source)
        self.assertIn("\nmbarrier.arrive.release.cta.shared::cta.b64 %s, [%a];\n", normalized)
        self.assertIn("\ntcgen05.wait::st.sync.aligned;\n", normalized)
        self.assertEqual(add_line_markers(normalized, "input.ptx")[1], 2)

    def test_rejects_ptx_without_an_entry_instruction(self) -> None:
        with self.assertRaisesRegex(ValueError, "no marked"):
            add_line_markers(".address_size 64\n", "empty.ptx")


class GuideOutputTest(unittest.TestCase):
    def test_output_collision_does_not_modify_input(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "guide"
            for suffix in (".ptx", ".input.ptx"):
                ptx = Path(f"{prefix}{suffix}")
                original = ".address_size 64\n.entry kernel() { ret; }\n"
                ptx.write_text(original)
                # No compiler may be invoked when an output aliases the input.
                with self.assertRaisesRegex(ValueError, "overwrite"):
                    generate_guide(ptx, prefix, ptxas=Path(sys.executable),
                                   nvdisasm=Path(sys.executable), arch="sm_80")
                self.assertEqual(ptx.read_text(), original)


class InstructionStreamTest(unittest.TestCase):
    def test_ignores_lineinfo_and_encoding_comments(self) -> None:
        disassembly = (
            '//## File "kernel.ptx", line 9\n'
            "        /*0010*/                   MOV R1, R2 ; /* 0x1 */\n"
            "                                              /* 0x2 */\n"
            '//## File "kernel.ptx", line 10\n'
            "        /*0020*/              @P0  BRA 0x40 ;\n"
        )

        self.assertEqual(
            instruction_stream(disassembly),
            ("/*0010*/                   MOV R1, R2 ;", "/*0020*/              @P0  BRA 0x40 ;"),
        )

    def test_empty_disassembly_has_no_instructions(self) -> None:
        self.assertEqual(instruction_stream("// no code\n"), ())


if __name__ == "__main__":
    unittest.main()
