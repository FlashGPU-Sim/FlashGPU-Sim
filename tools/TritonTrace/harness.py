from __future__ import annotations

import os
import re
import shutil
import subprocess
from importlib import resources
from pathlib import Path

from .session import KernelBinaryInfo, KernelLaunchInfo, TrackingSession


_TEMPLATE_TOKEN = re.compile(r"\{\{([A-Z0-9_]+)\}\}")


class HarnessGenerator:
    """Generate standalone CUDA launchers from one tracking session."""

    def __init__(self, session: TrackingSession):
        self.session = session

    @staticmethod
    def _load_template(name: str) -> str:
        return (
            resources.files("TritonTrace")
            .joinpath("templates")
            .joinpath(name)
            .read_text(encoding="utf-8")
        )

    @classmethod
    def _render_template(cls, name: str, **values) -> str:
        template = cls._load_template(name)
        required = set(_TEMPLATE_TOKEN.findall(template))
        missing = sorted(required - values.keys())
        if missing:
            raise ValueError(f"Template {name} is missing values for: {', '.join(missing)}")

        rendered = _TEMPLATE_TOKEN.sub(lambda match: str(values[match.group(1)]), template)
        unresolved = sorted(set(_TEMPLATE_TOKEN.findall(rendered)))
        if unresolved:
            raise ValueError(
                f"Template {name} still contains unresolved values: {', '.join(unresolved)}"
            )
        return rendered

    @staticmethod
    def _parse_ptx_signature(ptx_path: Path, kernel_name: str) -> int:
        """Count user parameters in a Triton PTX entry."""
        if not ptx_path.exists():
            return -1

        try:
            content = ptx_path.read_text()
            pattern = rf"\.visible\s+\.entry\s+{re.escape(kernel_name)}\s*\("
            match = re.search(pattern, content)
            if match is None:
                return -1

            closing_paren = content.find(")", match.end())
            if closing_paren == -1:
                return -1

            parameter_count = content[match.end() : closing_paren].count(".param")
            return max(0, parameter_count - 2)
        except Exception as error:
            print(f"    [WARNING] Could not parse PTX signature: {error}")
            return -1

    @staticmethod
    def _uses_dynamic_shared_memory(ptx_path: Path) -> bool:
        if not ptx_path.exists():
            return False
        try:
            return bool(re.search(r"\.extern\s+\.shared\s+.*\[\s*\]", ptx_path.read_text()))
        except Exception as error:
            print(f"    [WARNING] Could not check for dynamic shared memory: {error}")
            return False

    @staticmethod
    def _parse_ptx_target(ptx_path: Path) -> str:
        if ptx_path.exists():
            try:
                match = re.search(
                    r"^\s*\.target\s+([A-Za-z0-9_]+)",
                    ptx_path.read_text(),
                    re.MULTILINE,
                )
                if match:
                    return match.group(1)
            except Exception:
                pass
        return "sm_120a"

    @staticmethod
    def _parse_ptx_register_fallback(ptx_path: Path) -> int:
        if not ptx_path.exists():
            return 0
        try:
            total = 0
            declarations = re.findall(
                r"\.reg\s+\.([a-z0-9]+)\s+%[A-Za-z_][A-Za-z0-9_]*<(\d+)>",
                ptx_path.read_text(),
            )
            for register_type, count in declarations:
                count_value = int(count)
                if register_type == "b64":
                    total += count_value * 2
                elif register_type != "pred":
                    total += count_value
            return total
        except Exception:
            return 0

    def _generate_ptxinfo_sidecar(
        self,
        cubin_path: Path,
        ptx_path: Path,
        output_path: Path,
        kernel_name: str,
    ) -> None:
        """Write the resource sidecar consumed by FlashGPU-Sim."""
        arch = self._parse_ptx_target(ptx_path)
        registers = self._parse_ptx_register_fallback(ptx_path)
        stack = shared = local = constant_memory = global_memory = 0

        try:
            result = subprocess.run(
                ["cuobjdump", "--dump-resource-usage", str(cubin_path)],
                check=True,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
            global_match = re.search(r"\bGLOBAL:(\d+)", result.stdout)
            if global_match:
                global_memory = int(global_match.group(1))

            function_match = re.search(
                rf"Function\s+{re.escape(kernel_name)}:\s*\n\s*"
                r"REG:(\d+)\s+STACK:(\d+)\s+SHARED:(\d+)\s+LOCAL:(\d+)"
                r"\s+CONSTANT\[0\]:(\d+)",
                result.stdout,
            )
            if function_match:
                registers, stack, shared, local, constant_memory = map(
                    int, function_match.groups()
                )
        except Exception as error:
            print(
                "  Warning: cuobjdump resource usage failed, using PTX register "
                f"estimate: {error}"
            )

        output_path.write_text(
            f"ptxas info    : {global_memory} bytes gmem\n"
            f"ptxas info    : Compiling entry function '{kernel_name}' for '{arch}'\n"
            f"ptxas info    : Function properties for {kernel_name}\n"
            f"    {stack} bytes stack frame, 0 bytes spill stores, 0 bytes spill loads\n"
            f"ptxas info    : Used {registers} registers, {shared} bytes smem, "
            f"{constant_memory} bytes cmem[0], {local} bytes lmem\n"
        )
        print(f"  Generated ptxinfo sidecar: {output_path}")

    @staticmethod
    def _scratch_allocation_code(launch: KernelLaunchInfo) -> str:
        code = []
        total_ctas = launch.grid[0] * launch.grid[1] * launch.grid[2] * launch.num_ctas
        if launch.global_scratch_size > 0:
            code.append(
                f"""
    // Total size = per_cta_size * grid_size * num_ctas = per_cta * {total_ctas}
    cudaMalloc(&global_scratch, {launch.global_scratch_size});
    printf("  Allocated global_scratch: %zu bytes (alignment: %zu)\\n",
           (size_t){launch.global_scratch_size}, (size_t){launch.global_scratch_align});"""
            )
        if launch.profile_scratch_size > 0:
            code.append(
                f"""
    // Total size = per_cta_size * grid_size * num_ctas = per_cta * {total_ctas}
    cudaMalloc(&profile_scratch, {launch.profile_scratch_size});
    printf("  Allocated profile_scratch: %zu bytes (alignment: %zu)\\n",
           (size_t){launch.profile_scratch_size}, (size_t){launch.profile_scratch_align});"""
            )
        return "".join(code)

    @staticmethod
    def _scratch_cleanup_code(launch: KernelLaunchInfo) -> str:
        code = []
        if launch.global_scratch_size > 0:
            code.append(
                """
    cudaFree(global_scratch);
    printf("  Freed global_scratch\\n");"""
            )
        if launch.profile_scratch_size > 0:
            code.append(
                """
    cudaFree(profile_scratch);
    printf("  Freed profile_scratch\\n");"""
            )
        return "".join(code)

    def generate_kernel_template(self, kernel: KernelBinaryInfo) -> None:
        """Generate a generic launcher before launch arguments are known."""
        code = self._render_template(
            "kernel_template.cu.tpl",
            KERNEL_NAME=kernel.kernel_name,
            KERNEL_HASH=kernel.kernel_hash,
            NUM_WARPS=kernel.metadata.get("num_warps", 1),
            SHARED_MEMORY=kernel.metadata.get("shared", 0),
            BINARY_PATH=kernel.binary_path,
        )
        output_path = self.session.launchers_dir / (
            f"{kernel.kernel_name}_{kernel.kernel_hash[:8]}_template.cu"
        )
        output_path.write_text(code)
        print(f"  Generated template launcher: {output_path}")

    def generate_launch_harness(
        self,
        kernel: KernelBinaryInfo,
        launch: KernelLaunchInfo,
        *,
        validate_outputs: bool,
    ) -> None:
        """Generate one replay harness with captured launch arguments."""
        kernel_name = kernel.kernel_name
        launch_id = launch.launch_id
        binary_path = Path(kernel.binary_path)
        ptx_path = binary_path.parent / f"{kernel_name}.ptx"
        cubin_path = binary_path if binary_path.suffix == ".cubin" else None
        has_ptx = False
        if ptx_path.exists():
            try:
                ptx_path.read_text()
                has_ptx = True
            except Exception as error:
                print(f"  Warning: Could not read PTX: {error}")

        ptx_parameter_count = self._parse_ptx_signature(ptx_path, kernel_name)
        captured_argument_count = len(launch.args_info)
        if ptx_parameter_count >= 0 and ptx_parameter_count != captured_argument_count:
            raise RuntimeError(
                f"\nArgument count mismatch for kernel '{kernel_name}':\n"
                f"  Captured from Python: {captured_argument_count} arguments\n"
                f"  PTX kernel signature: {ptx_parameter_count} user parameters "
                "(+ 2 runtime scratch pointers)\n"
                f"  Mismatch: {captured_argument_count - ptx_parameter_count} arguments "
                "were optimized away by Triton\n\n"
                "  Triton optimizes away redundant parameters during compilation.\n"
                "  Cannot automatically generate the argument list without knowing which "
                "parameters were retained.\n"
                f"  Inspect the PTX file: {ptx_path}\n"
            )

        launchers_dir = self.session.launchers_dir.resolve()
        declarations = []
        loading_calls = []
        argument_pointers = []
        validation_calls = []
        cleanup = []

        for argument in launch.args_info:
            index = argument.index
            if argument.arg_type == "tensor" and argument.data_file:
                data_path = Path(argument.data_file)
                if not data_path.is_absolute():
                    data_path = (Path.cwd() / data_path).resolve()
                try:
                    relative_data_path = os.path.relpath(data_path, launchers_dir)
                except Exception:
                    relative_data_path = argument.data_file

                declarations.extend(
                    [
                        f"    void* d_arg{index};",
                        f"    size_t arg{index}_size = {argument.size_bytes};",
                    ]
                )
                loading_calls.append(
                    f'    d_arg{index} = load_tensor_arg(exe_path, "{relative_data_path}", '
                    f'arg{index}_size, {index}, "{argument.dtype}", '
                    f'"{list(argument.shape or ())}");'
                )
                if argument.size_bytes > 0:
                    loading_calls.append(f"    if (!d_arg{index}) return 1;")
                argument_pointers.append(f"&d_arg{index}")
                cleanup.append(f"    cudaFree(d_arg{index});")

                if validate_outputs and argument.output_file:
                    output_path = Path(argument.output_file)
                    if not output_path.is_absolute():
                        output_path = (Path.cwd() / output_path).resolve()
                    try:
                        relative_output_path = os.path.relpath(output_path, launchers_dir)
                    except Exception:
                        relative_output_path = argument.output_file
                    validation_calls.append(
                        f'    validate_tensor_output(d_arg{index}, exe_path, '
                        f'"{relative_output_path}", arg{index}_size, {index}, '
                        f'"{argument.dtype}");'
                    )
            elif argument.arg_type == "scalar":
                c_type = {
                    "int": "int32_t",
                    "float": "float",
                    "bool": "uint8_t",
                }.get(argument.dtype, "int32_t")
                value = argument.value
                declarations.append(f"    {c_type} arg{index} = {value};")
                loading_calls.append(
                    f'    printf("  Set arg[{index}]: {argument.dtype} = {value}\\n");'
                )
                argument_pointers.append(f"&arg{index}")

        if has_ptx:
            helper_functions = self._load_template("input_helpers.cu.inc")
            validation_include = ""
            validation_code = ""
            if validate_outputs:
                helper_functions += "\n" + self._load_template(
                    "validation_helpers.cu.inc"
                )
                validation_include = "#include <math.h>"
                validation_code = (
                    '\n    // Validate outputs\n    printf("\\nValidating outputs...\\n");\n'
                    + "\n".join(validation_calls)
                    + "\n"
                )

            shared_memory_config = '    printf("\\n");\n'
            if self._uses_dynamic_shared_memory(ptx_path) and launch.shared_memory > 0:
                shared_memory_config = f"""
    // Configure the opt-in dynamic shared-memory size.
    int shared_mem_bytes = {launch.shared_memory};
    CUresult attr_result = cuFuncSetAttribute(
        kernel_func,
        CU_FUNC_ATTRIBUTE_MAX_DYNAMIC_SHARED_SIZE_BYTES,
        shared_mem_bytes);
    if (attr_result != CUDA_SUCCESS) {{
        const char* errStr;
        cuGetErrorString(attr_result, &errStr);
        fprintf(stderr, "Warning: Failed to set shared memory size: %s\\n", errStr);
        fprintf(stderr, "Attempting to continue anyway...\\n");
    }} else {{
        printf("Configured dynamic shared memory: %d bytes\\n", shared_mem_bytes);
    }}
    printf("\\n");
"""

            fatbin_filename = f"{kernel_name}_launch{launch_id}_kernel.fatbin"
            code = self._render_template(
                "launch_harness.cu.tpl",
                KERNEL_NAME=kernel_name,
                LAUNCH_ID=launch_id,
                KERNEL_HASH=kernel.kernel_hash,
                VALIDATION_INCLUDE=validation_include,
                HELPER_FUNCTIONS=helper_functions,
                GRID=launch.grid,
                BLOCK=launch.block,
                SHARED_MEMORY=launch.shared_memory,
                FATBIN_FILENAME=fatbin_filename,
                SHARED_MEMORY_CONFIG=shared_memory_config,
                ARG_DECLARATIONS="\n".join(declarations),
                ARG_LOADING_CALLS="\n".join(loading_calls),
                SCRATCH_ALLOCATION=self._scratch_allocation_code(launch),
                ARG_POINTERS=", ".join(argument_pointers),
                GRID_X=launch.grid[0],
                GRID_Y=launch.grid[1],
                GRID_Z=launch.grid[2],
                BLOCK_X=launch.block[0],
                BLOCK_Y=launch.block[1],
                BLOCK_Z=launch.block[2],
                VALIDATION_CODE=validation_code,
                ARG_CLEANUP="\n".join(cleanup),
                SCRATCH_CLEANUP=self._scratch_cleanup_code(launch),
            )
        else:
            code = self._render_template("missing_ptx_harness.cu.tpl")

        harness_filename = f"{kernel_name}_launch{launch_id}_harness.cu"
        harness_path = self.session.launchers_dir / harness_filename
        harness_path.write_text(code)
        print(f"  Generated harness with arguments: {harness_path}")

        ptx_filename = f"{kernel_name}_launch{launch_id}_kernel.ptx"
        if has_ptx:
            copied_ptx_path = self.session.launchers_dir / ptx_filename
            shutil.copy(ptx_path, copied_ptx_path)
            print(f"  Copied PTX for inspection: {copied_ptx_path}")

        has_cubin = bool(cubin_path and cubin_path.exists())
        cubin_filename = f"{kernel_name}_launch{launch_id}_kernel.cubin"
        if has_cubin:
            copied_cubin_path = self.session.launchers_dir / cubin_filename
            shutil.copy(cubin_path, copied_cubin_path)
            print(f"  Copied CUBIN for linking: {copied_cubin_path}")
            ptxinfo_path = self.session.launchers_dir / (
                f"{kernel_name}_launch{launch_id}_kernel.ptxinfo"
            )
            self._generate_ptxinfo_sidecar(
                copied_cubin_path, ptx_path, ptxinfo_path, kernel_name
            )

        target_name = f"{kernel_name}_launch{launch_id}"
        makefile_path = self.session.launchers_dir / (
            f"{kernel_name}_launch{launch_id}_Makefile"
        )
        if has_ptx:
            fatbin_filename = f"{kernel_name}_launch{launch_id}_kernel.fatbin"
            dependencies = "$(CUBIN_FILE) $(PTX_FILE)" if has_cubin else "$(PTX_FILE)"
            legacy_images = "--image=profile=$(PTX_PROFILE),file=$(PTX_FILE)"
            image3_images = "--image3=kind=ptx,sm=$(ARCH_NUM),file=$(PTX_FILE)"
            if has_cubin:
                legacy_images = (
                    "--image=profile=$(ARCH),file=$(CUBIN_FILE) " + legacy_images
                )
                image3_images = (
                    "--image3=kind=elf,sm=$(ARCH_NUM),file=$(CUBIN_FILE) "
                    + image3_images
                )
            makefile = self._render_template(
                "launch.mk.tpl",
                KERNEL_NAME=kernel_name,
                LAUNCH_ID=launch_id,
                TARGET_NAME=target_name,
                PTX_FILENAME=ptx_filename,
                CUBIN_FILENAME=cubin_filename,
                FATBIN_FILENAME=fatbin_filename,
                FATBIN_DEPS=dependencies,
                FATBINARY_IMAGES_LEGACY=legacy_images,
                FATBINARY_IMAGES_IMAGE3=image3_images,
                HARNESS_FILENAME=harness_filename,
            )
        else:
            makefile = self._render_template(
                "fallback_launch.mk.tpl",
                KERNEL_NAME=kernel_name,
                LAUNCH_ID=launch_id,
                TARGET_NAME=target_name,
                HARNESS_FILENAME=harness_filename,
            )
        makefile_path.write_text(makefile)
        print(f"  Generated Makefile: {makefile_path}")
