"""Per-run decode-dump contracts, with CUDA tool calls mocked."""
import contextlib
import importlib.util
import io
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

TOOLS = Path(__file__).resolve().parents[3] / 'src/gpgpu-sim/flash/sass/tools'
SPEC = importlib.util.spec_from_file_location('sass_dump', TOOLS / 'dump_kernel_sassir.py')
DUMP = importlib.util.module_from_spec(SPEC)
sys.path.insert(0, str(TOOLS))
try:
    SPEC.loader.exec_module(DUMP)
    import cubit_to_sassir as CUBIT_EXPORT
finally:
    sys.path.pop(0)


class SassDumpTest(unittest.TestCase):
    def test_decoder_rejects_truncated_header_and_partial_parameter_bank(self):
        decoder = sys.modules['nvdisasm_to_sassir']
        self.assertIs(CUBIT_EXPORT.elf64_sections, decoder.elf64_sections)
        self.assertIs(CUBIT_EXPORT.instructions, decoder.raw_instructions)
        with self.assertRaisesRegex(ValueError, 'truncated ELF header'):
            decoder.elf64_sections(b'\x7fELF')
        heading = '.nv.info.k\n'
        records = (
            '\tEIATTR_CBANK_PARAM_SIZE\n\tValue: 0x0\n',
            '\tEIATTR_PARAM_CBANK\n\tValue: 0x0 0x210\n',
        )
        for record in records:
            with self.subTest(record=record), self.assertRaisesRegex(
                    RuntimeError, 'incomplete kernel parameter bank'):
                decoder.parse_kernel_abi(heading + record, 'k', 'sm90')
        self.assertEqual(decoder.parse_kernel_abi(heading, 'k', 'sm90'),
                         decoder.KernelAbi(0x210, 0, []))
        self.assertEqual(decoder.parse_kernel_abi(heading + ''.join(records),
                                                'k', 'sm90'),
                         decoder.KernelAbi(0x210, 0, []))

    def test_cli_resolves_tools_before_extraction_changes_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('binary', 'nvdisasm', 'cuobjdump'):
                (root / name).touch()
            args = SimpleNamespace(
                binary=Path(os.path.relpath(root / 'binary')),
                nvdisasm=Path(os.path.relpath(root / 'nvdisasm')),
                cuobjdump=Path(os.path.relpath(root / 'cuobjdump')),
                output=root / "k.sassir", kernel='k', fatbin_handle=1, arch='sm90')
            with patch.object(DUMP, 'parse_args', return_value=args), \
                 patch.object(DUMP, 'extract_kernel_cubin', return_value=root / 'k.cubin') as extract, \
                 patch.object(DUMP, 'dump_sassir', return_value=root / 'k.sassir') as populate, \
                 contextlib.redirect_stdout(io.StringIO()):
                self.assertEqual(DUMP.main(), 0)
            self.assertEqual(extract.call_args.args[0], root / 'cuobjdump')
            self.assertEqual(extract.call_args.args[5], root / 'nvdisasm')
            self.assertEqual(populate.call_args.args[3], root / 'nvdisasm')
            self.assertEqual(populate.call_args.args[4], root / 'cuobjdump')

    def test_failed_decode_never_publishes_or_leaves_temporary_output(self):
        def export_unknown(*args):
            args[-1].write('partial output')
            return 3, 1
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            with patch.object(DUMP, 'export', side_effect=export_unknown):
                with self.assertRaises(RuntimeError):
                    DUMP.dump_sassir(root / 'k', 'k', 'sm90', root / 'nv', root / 'cu', root / 'k.sassir')
            self.assertFalse(any(p.is_file() for p in root.rglob('*')))

    def test_existing_dump_is_regenerated_not_reused(self):
        def export_known(*args):
            args[-1].write('fresh decode')
            return 3, 0
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            destination = root / 'k.sassir'
            destination.write_text('existing decode')
            with patch.object(DUMP, 'export', side_effect=export_known) as export:
                result = DUMP.dump_sassir(root / 'k', 'k', 'sm90', root / 'nv', root / 'cu', destination)
            self.assertEqual(result, destination)
            export.assert_called_once()
            self.assertEqual(destination.read_text(), 'fresh decode')


if __name__ == '__main__':
    unittest.main()
