"""Frontend isolation tests; no CUDA tools or simulator process required."""
import os
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest
from unittest.mock import patch

from runner.executors import TestExecutors
from runner.cli import load_settings, parse_arguments
from runner.model import Settings
from runner.gtest import GTest
from runner.errors import DiscoveryError


class FrontendEnvironmentTest(unittest.TestCase):
    def test_discovery_driver_stub_is_local_and_errors_are_visible(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binary = root / 'tests'
            binary.touch(mode=0o755)
            stub = root / 'cuda/lib64/stubs/libcuda.so'
            stub.parent.mkdir(parents=True)
            stub.touch()
            with patch.dict(os.environ, {
                'CUDA_INSTALL_PATH': str(root / 'cuda'),
                'GPGPUSIM_ROOT': str(root),
                'LD_LIBRARY_PATH': str(root / 'lib/release'),
                'LD_PRELOAD': '/inherited/simulator.so',
            }, clear=True), patch('runner.gtest.subprocess.run') as run:
                run.return_value = SimpleNamespace(
                    returncode=0, stdout='Suite.\n  Case\n', stderr='')
                self.assertEqual(GTest(None, 10).list_cases(binary, root),
                                 ['Suite.Case'])
                env = run.call_args.kwargs['env']
                self.assertEqual(env['LD_PRELOAD'], str(stub))
                self.assertEqual(env['LD_LIBRARY_PATH'], str(root / 'cuda/lib64'))
                self.assertEqual(os.environ['LD_PRELOAD'], '/inherited/simulator.so')
                stub.unlink()
                run.return_value = SimpleNamespace(
                    returncode=127, stdout='', stderr='missing libcuda.so.1')
                with self.assertRaisesRegex(DiscoveryError, 'missing libcuda.so.1'):
                    GTest(None, 10).list_cases(binary, root)
                self.assertNotIn('LD_PRELOAD', run.call_args.kwargs['env'])

    def executor(self, root, settings):
        return TestExecutors(root / 'tests', settings, None, None, None, None)

    def test_ptx_drops_all_inherited_sass_settings(self):
        with patch.dict(os.environ, {
            'FLASHGPU_SASS_AUTO': '1', 'FLASHGPU_SASS_TIMING': '1',
            'FLASHGPU_SASS_IR': '/wrong.sassir',
            'FLASHGPU_SASS_MAX_INSTRUCTIONS_PER_CTA': '1',
            'OMP_NUM_THREADS': '4',
            'PTX_SIM_MODE_FUNC': '1',
        }, clear=True):
            env = self.executor(Path('/repo'), Settings())._gtest_environment(None, Path('/binary'))
        self.assertFalse(any(k.startswith('FLASHGPU_SASS_') for k in env))
        self.assertEqual(env['OMP_NUM_THREADS'], '4')
        self.assertNotIn('PTX_SIM_MODE_FUNC', env)

    def test_explicit_sassir_does_not_inherit_timing_or_auto(self):
        with patch.dict(os.environ, {'FLASHGPU_SASS_AUTO': '1', 'FLASHGPU_SASS_TIMING': '1'}, clear=True):
            env = self.executor(Path('/repo'), Settings(sassir='/selected.sassir'))._gtest_environment(None, Path('/binary'))
        self.assertEqual(env['FLASHGPU_SASS_IR'], '/selected.sassir')
        self.assertNotIn('FLASHGPU_SASS_AUTO', env)
        self.assertNotIn('FLASHGPU_SASS_TIMING', env)

    def test_auto_functional_and_timing_are_explicit(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for name in ('cuda/bin/nvdisasm', 'cuda/bin/cuobjdump', 'src/gpgpu-sim/flash/sass/tools/dump_kernel_sassir.py'):
                path = root / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.touch()
            for timing in (False, True):
                with self.subTest(timing=timing), patch.dict(os.environ, {
                    'CUDA_INSTALL_PATH': str(root / 'cuda'),
                    'FLASHGPU_SASS_IR': '/wrong.sassir',
                    'FLASHGPU_SASS_TIMING': '1',
                }, clear=True):
                    selection = SimpleNamespace(architecture=SimpleNamespace(name='sm90'))
                    settings = load_settings(parse_arguments([
                        'run', '--sass-timing' if timing else '--sass',
                    ]))
                    env = self.executor(root, settings)._gtest_environment(selection, root / 'binary')
                self.assertNotIn('FLASHGPU_SASS_IR', env)
                self.assertEqual(env['FLASHGPU_SASS_AUTO'], '1')
                self.assertEqual(env.get('FLASHGPU_SASS_TIMING'), '1' if timing else None)
                self.assertEqual(env['FLASHGPU_SASS_BINARY'], str(root / 'binary'))


if __name__ == '__main__':
    unittest.main()
