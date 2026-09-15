"""Guard compilation/link source agreement without CUDA or a populated build tree."""
from pathlib import Path
import subprocess
import unittest


ROOT = Path(__file__).resolve().parents[3]
SASS = Path('src/gpgpu-sim/flash/sass')


class SassBuildSourcesTest(unittest.TestCase):
    def sources(self, cwd, include, variable='SASS_SRCS'):
        result = subprocess.run(
            ['make', '--no-print-directory', '-f', '-', 'print'], cwd=cwd,
            input=f'include {include}\nprint:\n\t@echo $({variable})\n',
            text=True, capture_output=True, check=True)
        return {(cwd / name).resolve() for name in result.stdout.split()}

    def test_root_link_and_subdirectory_compile_use_same_sources(self):
        linked = self.sources(ROOT, SASS / 'sources.mk')
        compiled = self.sources(ROOT / 'src/gpgpu-sim',
                                'flash/sass/sources.mk')
        self.assertEqual(linked, compiled)
        self.assertIn(ROOT / SASS / 'runtime/runtime_adapter.cc', linked)
        self.assertIn(ROOT / SASS / 'functional/integer.cc', linked)
        self.assertIn(ROOT / SASS / 'frontend.cc', linked)
        self.assertFalse(any('tools' in path.parts for path in linked))
        frontend = Path('src/gpgpu-sim/flash/frontend')
        linked_frontend = self.sources(ROOT, frontend / 'sources.mk',
                                       'FRONTEND_SRCS')
        self.assertEqual(linked_frontend, self.sources(
            ROOT / 'src/gpgpu-sim', 'flash/frontend/sources.mk',
            'FRONTEND_SRCS'))
        self.assertEqual(linked_frontend, {
            ROOT / frontend / 'execution_frontend.cc',
            ROOT / frontend / 'ptx_shader_adapter.cc'})

    def test_all_shared_library_targets_use_source_derived_objects(self):
        makefile = (ROOT / 'Makefile').read_text()
        self.assertNotIn('$(SIM_OBJ_FILES_DIR)/gpgpu-sim/flash/sass/*.o', makefile)
        self.assertEqual(makefile.count('$(SASS_OBJECTS) \\'), 3)
        self.assertEqual(makefile.count('$(FRONTEND_OBJECTS) \\'), 3)
        self.assertIn('$(patsubst src/%.cc,$(SIM_OBJ_FILES_DIR)/%.o,$(SASS_SRCS))',
                      makefile)
