"""Module constant banks and CUDA device-global relocation contracts."""
import importlib.util
from pathlib import Path
import struct
import unittest

TOOLS = Path(__file__).resolve().parents[3] / 'src/gpgpu-sim/flash/sass/tools'
SPEC = importlib.util.spec_from_file_location('constant_globals_export', TOOLS / 'nvdisasm_to_sassir.py')
EXPORT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(EXPORT)


def fixture(reloc_type=2, addend=8, symbol_section=4, global_type=8, offset=0, initialized_local=False, external=None):
    names = ['', '.shstrtab', '.strtab', '.symtab', '.nv.global',
             '.nv.constant4', '.rela.nv.constant4', '.nv.constant2.k']
    if initialized_local:
        names[4] = '.nv.global.init'
    shstrings = b'\0'
    name_offsets = []
    for name in names:
        name_offsets.append(len(shstrings))
        shstrings += name.encode() + b'\0'
    symbols = bytes(24) + struct.pack('<IBBHQQ', 1, 0x01 if initialized_local else 0x11, 0, symbol_section, 0, 32)
    bodies = [b'', shstrings, b'\0state\0', symbols, bytes(32), bytes(8),
              struct.pack('<QQq', offset, (1 << 32) | reloc_type, addend), b'\x12\x34']
    if external:
        bodies[2] = b'\0' + external.encode() + b'\0'
        bodies[3] = bytes(24) + struct.pack('<IBBHQQ', 1, 0x12, 0, 0, 0, 0)
        bodies[6] = struct.pack('<QQq', offset, (1 << 32) | reloc_type, 0)
    data = bytearray(64 + len(names) * 64)
    data[:6] = b'\x7fELF\x02\x01'
    struct.pack_into('<Q', data, 0x28, 64)
    struct.pack_into('<HHH', data, 0x3a, 64, len(names), 1)
    types = [0, 3, 3, 2, global_type, 1, 4, 1]
    for i, body in enumerate(bodies):
        struct.pack_into('<IIQQQQIIQQ', data, 64 + i * 64,
                         name_offsets[i], types[i], 0, 0, len(data), len(body),
                         2 if i == 3 else 3 if i == 6 else 0,
                         5 if i == 6 else 0, 8, 24 if i in (3, 6) else 0)
        data += body
    return bytes(data)


class ConstantGlobalsTest(unittest.TestCase):
    def test_module_and_kernel_banks_and_symbol_addend(self):
        data = fixture()
        sections = EXPORT.elf64_sections(data)
        self.assertEqual(EXPORT.find_kernel_constant_banks(data, sections, 'k'),
                         [(2, b'\x12\x34'), (4, bytes(8))])
        self.assertEqual(list(EXPORT.constant_global_relocations(data, sections, 'k')),
                         [(4, 0, 'state', 32, 8)])
        # The module bank is also available to another kernel in the same image.
        self.assertEqual(EXPORT.find_kernel_constant_banks(data, sections, 'other'),
                         [(4, bytes(8))])

    def test_initialized_local_payload_and_addend(self):
        data = fixture(global_type=1, initialized_local=True)
        self.assertEqual(list(EXPORT.constant_global_relocations(
            data, EXPORT.elf64_sections(data), 'k')),
            [(4, 0, 'state', 32, 8, bytes(32).hex())])

    def test_unresolved_external_functions_are_not_data_objects(self):
        for name in ('__assertfail', 'vprintf', '__cuda_syscall_cp_async_bulk_tensor_2d_tile_unicast'):
            data = fixture(external=name)
            self.assertEqual(list(EXPORT.constant_global_relocations(
                data, EXPORT.elf64_sections(data), 'k')), [])

    def test_unsupported_or_invalid_relocations_fail_closed(self):
        for args in [dict(reloc_type=1), dict(addend=-1), dict(addend=33),
                     dict(symbol_section=0), dict(global_type=1), dict(offset=1)]:
            with self.subTest(args=args):
                data = fixture(**args)
                with self.assertRaises(ValueError):
                    list(EXPORT.constant_global_relocations(data, EXPORT.elf64_sections(data), 'k'))


if __name__ == '__main__':
    unittest.main()
