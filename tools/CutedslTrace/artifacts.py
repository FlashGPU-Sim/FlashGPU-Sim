"""Export the exact captured TVM FFI module and its device artifacts."""
import hashlib
import re
from pathlib import Path


def export_module(compiled, directory, name, target):
    directory.mkdir(parents=True, exist_ok=True)
    compiled.export_to_c(str(directory / f"{name}.o"), function_name=name)
    for extension, content in (("ptx", compiled.__ptx__), ("cubin", compiled.__cubin__)):
        if content is None:
            matches = [p for p in (directory.parent / "dump").glob(f"*.{extension}")
                       if compiled.function_name in p.name]
            if len(matches) != 1:
                raise RuntimeError(f"Expected one {extension} for {compiled.function_name}, found {matches}")
            content = matches[0].read_bytes()
        if isinstance(content, str) and "\n" not in content and Path(content).is_file():
            data = Path(content).read_bytes()
        else:
            data = content.encode() if isinstance(content, str) else bytes(content)
        (directory / f"{name}.{extension}").write_bytes(data)
    text = (directory / f"{name}.ptx").read_text()
    match = re.search(r"^\s*\.target\s+(\w+)", text, re.M)
    if match is None or match[1] != target:
        raise RuntimeError(f"Captured PTX does not target {target}: {text[:300]!r}")
    return {p.suffix[1:]: {"path": f"modules/{p.name}",
                          "sha256": hashlib.sha256(p.read_bytes()).hexdigest()}
            for p in directory.glob(name + ".*")}
