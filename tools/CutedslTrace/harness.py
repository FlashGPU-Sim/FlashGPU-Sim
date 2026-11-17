"""Generate a TVM FFI replay from module, argument and storage records."""
import importlib.metadata
import json
from pathlib import Path


def tensor_declaration(name, tensor):
    dims = len(tensor["shape"])
    shape = ", ".join(map(str, tensor["shape"])) or "1"
    strides = ", ".join(map(str, tensor["strides"])) or "1"
    return f'''int64_t {name}_shape[] = {{{shape}}}, {name}_strides[] = {{{strides}}};
    DLTensor {name}{{static_cast<char*>(buffers[{tensor['storage']}]) + {tensor['offset']},
        {{kDLCUDA, 0}}, {dims}, {{{tensor['code']}, {tensor['bits']}, 1}},
        {name}_shape, {name}_strides, 0}};'''


def render(session):
    templates = Path(__file__).parent / "templates"
    declarations, body = [], []
    for module in session.modules:
        declarations.append(f'extern "C" int __tvm_ffi_{module["name"]}(void*, const TVMFFIAny*, int32_t, TVMFFIAny*);')
    for index, storage in enumerate(session.storages):
        body.append(f'load_buffer(root / {json.dumps(storage["path"])}, {storage["bytes"]}, buffers[{index}]);')
    for index, call in enumerate(session.calls):
        module = session.modules[call["module"]]["name"]
        body.append('{')
        body.append(f'TVMFFIAny args[{max(1, len(call["arguments"]))}]{{}}, result{{}};')
        for arg_index, argument in enumerate(call["arguments"]):
            name = f"tensor{arg_index}"
            kind = argument["kind"]
            if kind == "tensor":
                body.append(tensor_declaration(name, argument))
                body.append(f'args[{arg_index}].type_index = kTVMFFIDLTensorPtr; args[{arg_index}].v_ptr = &{name};')
            elif kind != "none":
                tag = {"bool": "kTVMFFIBool", "int": "kTVMFFIInt", "float": "kTVMFFIFloat"}[kind]
                field = "v_float64" if kind == "float" else "v_int64"
                value = repr(argument["value"]) if kind == "float" else str(int(argument["value"]))
                body.append(f'args[{arg_index}].type_index = {tag}; args[{arg_index}].{field} = {value};')
        body.append(f'''const auto ptx = root / "modules/{module}.ptx";
        if (setenv("GPGPUSIM_CUDA_LIBRARY_PTX", ptx.c_str(), 1)) throw std::runtime_error("setenv failed");
        ffi_check(__tvm_ffi_{module}(nullptr, args, {len(call['arguments'])}, &result));
        cuda_check(cudaDeviceSynchronize());
        std::printf("Replay call {index}: PASS\\n");
        }}''')
    for index, storage in enumerate(session.storages):
        body.append(f'save_buffer(root / "data/storage{index}.result.bin", buffers[{index}], {storage["bytes"]});')
    for index, check in enumerate(session.checks):
        name = f"check{index}"
        body.append(tensor_declaration(name, check["tensor"]))
        body.append(f'check_tensor({name}, root / {json.dumps(check["expected"])});')
    source = (templates / "harness.cc.tpl").read_text()
    for key, value in {"@DECLARATIONS@": "\n".join(declarations),
                       "@BODY@": "\n".join(body),
                       "@BUFFER_COUNT@": str(len(session.storages))}.items():
        source = source.replace(key, value)
    (session.output / "harness.cc").write_text(source)
    import tvm_ffi
    tvm_root = Path(tvm_ffi.__file__).parent
    # Distribution root contains the DSL's lib directory, outside its Python package.
    runtime = Path(importlib.metadata.distribution("nvidia-cutlass-dsl-libs-base").locate_file("nvidia_cutlass_dsl/lib"))
    if not (runtime / "libcuda_dialect_runtime_static.a").exists():
        raise RuntimeError("CuTe DSL static runtime was not found")
    makefile = (templates / "Makefile.tpl").read_text()
    for key, path in {"@TVM_ROOT@": str(tvm_root), "@DSL_LIB@": str(runtime)}.items():
        if any(c in path for c in "\n\r\t#$ `"):
            raise ValueError("Build dependency paths cannot contain whitespace or Make metacharacters")
        makefile = makefile.replace(key, path)
    (session.output / "Makefile").write_text(makefile)
