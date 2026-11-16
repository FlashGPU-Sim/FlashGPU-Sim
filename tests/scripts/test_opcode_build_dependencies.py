#!/usr/bin/env python3
"""Exercise real make/makedepend rules without a CUDA build or source mutation."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]

class OpcodeBuildDependencies(unittest.TestCase):
    def test_generated_parser_objects_rebuild_after_opcode_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            src = root / 'src/cuda-sim'
            out = root / 'objects/cuda-sim'
            src.mkdir(parents=True)
            out.mkdir(parents=True)
            shutil.copy2(ROOT / 'version_detection.mk', root / 'version_detection.mk')
            for name in ['Makefile', 'ptx.y', 'ptx.l', 'opcodes.h', 'opcodes.def']:
                shutil.copy2(ROOT / 'src/cuda-sim' / name, src / name)
            (src / 'placeholder.cc').touch()
            (out / 'Makefile.makedepend').touch()
            stamp = time.time() + 100
            for stem in ['ptx.tab', 'lex.ptx_']:
                for ext in ['c', 'o']:
                    file = out / (stem + '.' + ext)
                    file.touch()
                    os.utime(file, (stamp, stamp))
            for header in ['opcodes.h', 'opcodes.def']:
                with self.subTest(header=header):
                    os.utime(src / header, (stamp + 10, stamp + 10))
                    for stem in ['ptx.tab', 'lex.ptx_']:
                        result = subprocess.run([
                            'make', '-nr', '-o', 'depend', str(out / (stem + '.o')),
                            'SIM_OBJ_FILES_DIR=' + str(out.parent), 'CPP=echo',
                            'CUDA_INSTALL_PATH=' + os.environ.get('CUDA_INSTALL_PATH', '/usr/local/cuda'),
                        ], cwd=src, text=True, capture_output=True)
                        self.assertEqual(result.returncode, 0, result.stderr)
                        self.assertIn(' -c ', result.stdout, result.stdout)
                        self.assertIn(str(out / (stem + '.c')), result.stdout)
                    os.utime(src / header, (stamp - 10, stamp - 10))

    def test_flash_handler_transitive_opcode_dependencies(self):
        if not shutil.which('makedepend'):
            self.skipTest('makedepend is required by the simulator build')
        with tempfile.TemporaryDirectory() as tmp:
            for name in ['elect', 'ld_st_matrix', 'mbarrier', 'tensormap', 'tma']:
                with self.subTest(handler=name):
                    dep = Path(tmp) / (name + '.mk')
                    dep.touch()
                    result = subprocess.run([
                        'makedepend', '-f' + str(dep), 'flash/' + name + '.cc'
                    ], cwd=ROOT / 'src/gpgpu-sim', text=True, capture_output=True)
                    self.assertEqual(result.returncode, 0, result.stderr)
                    text = dep.read_text()
                    self.assertIn('opcodes.h', text)
                    self.assertIn('opcodes.def', text)

if __name__ == '__main__':
    unittest.main()
