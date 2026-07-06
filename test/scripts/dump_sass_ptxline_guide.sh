#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TEST_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
ROOT_DIR="$(cd "${TEST_DIR}/.." && pwd)"

CUDA_INSTALL_PATH="${CUDA_INSTALL_PATH:-/usr/local/cuda-12.8}"
export PATH="${CUDA_INSTALL_PATH}/bin:${PATH}"

BINARY=""
ARCH=""
OUT_DIR=""
LABEL=""
CUBIN_PATTERN=""
FUNCTION_EXACT=""
FUNCTION_REGEX=""
LIST_FUNCTIONS=0
FORCE=0

usage() {
  cat <<'EOF'
Usage:
  dump_sass_ptxline_guide.sh --binary PATH --arch sm_90a --out-dir DIR [options]

Required:
  --binary PATH          CUDA test binary or executable containing the target cubin
  --arch ARCH            Cubin arch suffix, for example sm_90a or sm_120a
  --out-dir DIR          Output directory

Options:
  --label NAME           Output file prefix. Default: basename(binary)
  --cubin-pattern GLOB   Filter extracted cubin basename, for binaries with multiple cubins
  --function NAME        Select exactly one .text.<NAME> function section
  --function-regex REGEX Select exactly one .text function section by Python regex
  --list-functions       Only require full dump and function list; selection is optional
  --force                Overwrite this script's output files/directories under --out-dir
  -h, --help             Show this help

Outputs:
  DIR/LABEL.ARCH.cubin
  DIR/LABEL.ARCH.ptxline.full.sass
  DIR/LABEL.ARCH.functions.txt
  DIR/LABEL.ARCH.ptxline.selected.sass    when --function/--function-regex is used
  DIR/LABEL.ARCH.metadata.txt

Pass the selected .ptxline.sass to run_sim_queue.py through the job TSV
sass_ptxline column. The runner copies it into each job cwd using the simulator's
fixed guide filename.
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --binary)
      BINARY="$2"
      shift 2
      ;;
    --arch)
      ARCH="$2"
      shift 2
      ;;
    --out-dir)
      OUT_DIR="$2"
      shift 2
      ;;
    --label)
      LABEL="$2"
      shift 2
      ;;
    --cubin-pattern)
      CUBIN_PATTERN="$2"
      shift 2
      ;;
    --function)
      FUNCTION_EXACT="$2"
      shift 2
      ;;
    --function-regex)
      FUNCTION_REGEX="$2"
      shift 2
      ;;
    --list-functions)
      LIST_FUNCTIONS=1
      shift
      ;;
    --force)
      FORCE=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "${BINARY}" || -z "${ARCH}" || -z "${OUT_DIR}" ]]; then
  usage >&2
  exit 2
fi
if [[ -n "${FUNCTION_EXACT}" && -n "${FUNCTION_REGEX}" ]]; then
  echo "--function and --function-regex are mutually exclusive" >&2
  exit 2
fi
if [[ ! -x "${BINARY}" && ! -f "${BINARY}" ]]; then
  echo "missing binary: ${BINARY}" >&2
  exit 1
fi
if ! command -v cuobjdump >/dev/null 2>&1; then
  echo "cuobjdump not found in PATH; set CUDA_INSTALL_PATH if needed" >&2
  exit 1
fi
if ! command -v nvdisasm >/dev/null 2>&1; then
  echo "nvdisasm not found in PATH; set CUDA_INSTALL_PATH if needed" >&2
  exit 1
fi

BINARY_ABS="$(readlink -f "${BINARY}")"
if [[ -z "${LABEL}" ]]; then
  LABEL="$(basename "${BINARY_ABS}")"
fi

mkdir -p "${OUT_DIR}"
OUT_DIR_ABS="$(readlink -f "${OUT_DIR}")"
EXTRACT_DIR="${OUT_DIR_ABS}/${LABEL}.${ARCH}.extract"
CUBIN_OUT="${OUT_DIR_ABS}/${LABEL}.${ARCH}.cubin"
FULL_SASS="${OUT_DIR_ABS}/${LABEL}.${ARCH}.ptxline.full.sass"
FUNCTIONS_TXT="${OUT_DIR_ABS}/${LABEL}.${ARCH}.functions.txt"
SELECTED_SASS="${OUT_DIR_ABS}/${LABEL}.${ARCH}.ptxline.selected.sass"
METADATA="${OUT_DIR_ABS}/${LABEL}.${ARCH}.metadata.txt"
CUOBJDUMP_LOG="${OUT_DIR_ABS}/${LABEL}.${ARCH}.cuobjdump-xelf.log"
NVDISASM_LOG="${OUT_DIR_ABS}/${LABEL}.${ARCH}.nvdisasm.log"

guard_path() {
  local path="$1"
  if [[ -e "${path}" && "${FORCE}" != "1" ]]; then
    echo "refusing to overwrite ${path}; pass --force or choose a new --out-dir/--label" >&2
    exit 1
  fi
}

guard_path "${EXTRACT_DIR}"
guard_path "${CUBIN_OUT}"
guard_path "${FULL_SASS}"
guard_path "${FUNCTIONS_TXT}"
guard_path "${METADATA}"
if [[ -n "${FUNCTION_EXACT}" || -n "${FUNCTION_REGEX}" ]]; then
  guard_path "${SELECTED_SASS}"
fi

if [[ "${FORCE}" == "1" ]]; then
  rm -rf -- "${EXTRACT_DIR}"
  rm -f -- "${CUBIN_OUT}" "${FULL_SASS}" "${FUNCTIONS_TXT}" \
    "${SELECTED_SASS}" "${METADATA}" "${CUOBJDUMP_LOG}" "${NVDISASM_LOG}"
fi

mkdir -p "${EXTRACT_DIR}"
(
  cd "${EXTRACT_DIR}"
  cuobjdump -xelf all "${BINARY_ABS}" >"${CUOBJDUMP_LOG}" 2>&1
)

mapfile -t CUBINS < <(
  find "${EXTRACT_DIR}" -maxdepth 1 -type f \
    \( -name "*.${ARCH}.cubin" -o -name "*${ARCH}.cubin" \) \
    -printf '%p\n' | sort
)

if [[ -n "${CUBIN_PATTERN}" ]]; then
  FILTERED=()
  for cubin in "${CUBINS[@]}"; do
    base="$(basename "${cubin}")"
    case "${base}" in
      ${CUBIN_PATTERN}) FILTERED+=("${cubin}") ;;
    esac
  done
  CUBINS=("${FILTERED[@]}")
fi

if [[ "${#CUBINS[@]}" -ne 1 ]]; then
  {
    echo "expected exactly one ${ARCH} cubin, found ${#CUBINS[@]}"
    printf '  %s\n' "${CUBINS[@]}"
    echo "use --cubin-pattern if the binary contains multiple cubins"
  } >&2
  exit 1
fi

cp -f "${CUBINS[0]}" "${CUBIN_OUT}"
nvdisasm -gp --print-code "${CUBIN_OUT}" >"${FULL_SASS}" 2>"${NVDISASM_LOG}"

python3 - "${FULL_SASS}" "${FUNCTIONS_TXT}" "${FUNCTION_EXACT}" \
  "${FUNCTION_REGEX}" "${SELECTED_SASS}" "${LIST_FUNCTIONS}" <<'PY'
import re
import sys
from pathlib import Path

full_path = Path(sys.argv[1])
functions_path = Path(sys.argv[2])
exact = sys.argv[3]
regex_text = sys.argv[4]
selected_path = Path(sys.argv[5])
list_only = sys.argv[6] == "1"

text = full_path.read_text(errors="replace").splitlines(keepends=True)
section_re = re.compile(r"\.text\.([^\s,\"':()]+)")
sections = []
current = None

for index, line in enumerate(text):
    match = section_re.search(line)
    if not match:
        continue
    name = match.group(1)
    if current is not None and current["name"] == name:
        continue
    if current is not None:
        current["end"] = index
        sections.append(current)
    current = {"name": name, "start": index, "end": len(text)}

if current is not None:
    sections.append(current)

dedup = []
seen = set()
for section in sections:
    name = section["name"]
    if name not in seen:
        seen.add(name)
        dedup.append(name)
functions_path.write_text("\n".join(dedup) + ("\n" if dedup else ""))

if not exact and not regex_text:
    if not list_only:
        print("no --function/--function-regex provided; full dump and function list emitted", file=sys.stderr)
    sys.exit(0)

if exact:
    matches = [s for s in sections if s["name"] == exact]
else:
    regex = re.compile(regex_text)
    matches = [s for s in sections if regex.search(s["name"])]

if len(matches) != 1:
    print(f"expected exactly one matching .text section, found {len(matches)}", file=sys.stderr)
    for section in matches[:50]:
        print(f"  {section['name']}", file=sys.stderr)
    if len(matches) > 50:
        print(f"  ... {len(matches) - 50} more", file=sys.stderr)
    sys.exit(1)

section = matches[0]
selected_path.write_text("".join(text[section["start"]:section["end"]]))
PY

{
  echo "root=${ROOT_DIR}"
  echo "binary=${BINARY_ABS}"
  echo "arch=${ARCH}"
  echo "label=${LABEL}"
  echo "cubin=${CUBIN_OUT}"
  echo "full_ptxline_sass=${FULL_SASS}"
  echo "functions=${FUNCTIONS_TXT}"
  if [[ -n "${FUNCTION_EXACT}" || -n "${FUNCTION_REGEX}" ]]; then
    echo "selected_ptxline_sass=${SELECTED_SASS}"
  fi
  echo "function=${FUNCTION_EXACT}"
  echo "function_regex=${FUNCTION_REGEX}"
  echo "cuda_install_path=${CUDA_INSTALL_PATH}"
  echo "cuobjdump=$(command -v cuobjdump)"
  echo "nvdisasm=$(command -v nvdisasm)"
} >"${METADATA}"

echo "cubin: ${CUBIN_OUT}"
echo "full ptxline sass: ${FULL_SASS}"
echo "functions: ${FUNCTIONS_TXT}"
if [[ -n "${FUNCTION_EXACT}" || -n "${FUNCTION_REGEX}" ]]; then
  echo "selected ptxline sass: ${SELECTED_SASS}"
fi
echo "metadata: ${METADATA}"
