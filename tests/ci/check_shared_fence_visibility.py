#!/usr/bin/env python3
"""Exercise the real fence path with an intentionally visible store delay."""
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile

root = Path(__file__).resolve().parents[2]
parent = root / "tests/logs/ci/shared-fence-visibility"
parent.mkdir(parents=True, exist_ok=True)
work = Path(tempfile.mkdtemp(prefix="visibility-", dir=parent))
shutil.copytree(root / "configs/SM100_B200", work, dirs_exist_ok=True)
with (work / "gpgpusim.config").open("a") as config:
    config.write("\n-gpgpu_execution_frontend ptx\n"
                 "-gpgpu_ptx_reorder 0\n-gpgpu_ptx_reorder_sass_guided 0\n"
                 "-gpgpu_smem_store_visibility_latency 512\n")
env = {k: v for k, v in os.environ.items()
       if not k.startswith("FLASHGPU_SASS_") and k != "GTEST_OUTPUT"}
env.setdefault("OMP_NUM_THREADS", "2")
command = [str(root / "tests/build/bin/sm100/integration_tests"),
           "--gtest_filter=SharedFenceVisibility.*"]
with (work / "run.log").open("w") as output:
    result = subprocess.run(command, cwd=work, env=env, stdout=output,
                            stderr=subprocess.STDOUT, timeout=300)
log = (work / "run.log").read_text(errors="replace")
if (result.returncode or not re.search(r"\[\s*PASSED\s*\] 2 tests\.", log)
        or re.search(r"\[\s*SKIPPED\s*\]", log)):
    print(log[-16000:])
    raise SystemExit(result.returncode or 1)
print(f"Both numerical fence launches respect the 512-cycle visibility bound: {work / 'run.log'}")
