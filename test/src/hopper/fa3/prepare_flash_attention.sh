#!/usr/bin/env bash
set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

flash_attention_dir="${FLASH_ATTENTION_DIR:-${script_dir}/flash-attention}"
flash_attention_repo="${FLASH_ATTENTION_REPO:-https://github.com/Dao-AILab/flash-attention.git}"
flash_attention_base_commit="${FLASH_ATTENTION_BASE_COMMIT:-d80a77103021c4e980f8cbbf85774f6a19e6474a}"
cutlass_commit="${FLASH_ATTENTION_CUTLASS_COMMIT:-7127592069c2fe01b041e174ba4345ef9b279671}"
force_reset="${FLASH_ATTENTION_FORCE:-0}"

patches=(
  "${script_dir}/patches/flash-attention-fa3-profile-hooks.patch"
  "${script_dir}/patches/flash-attention-fa2-fa3-sensitivity-hooks.patch"
)

die() {
  echo "error: $*" >&2
  exit 1
}

ensure_commit() {
  local repo_dir="$1"
  local commit="$2"
  local remote="${3:-origin}"

  if git -C "${repo_dir}" cat-file -e "${commit}^{commit}" 2>/dev/null; then
    return 0
  fi
  git -C "${repo_dir}" fetch --tags "${remote}" || true
  git -C "${repo_dir}" cat-file -e "${commit}^{commit}" 2>/dev/null ||
    die "commit ${commit} is not available in ${repo_dir}"
}

is_git_checkout() {
  local repo_dir="$1"
  git -C "${repo_dir}" rev-parse --is-inside-work-tree >/dev/null 2>&1
}

tree_is_clean() {
  local repo_dir="$1"
  git -C "${repo_dir}" diff --quiet &&
    git -C "${repo_dir}" diff --cached --quiet
}

patches_are_applied() {
  local repo_dir="$1"
  local patch
  for patch in "${patches[@]}"; do
    git -C "${repo_dir}" apply --reverse --check "${patch}" >/dev/null 2>&1 ||
      return 1
  done
}

if [[ -e "${flash_attention_dir}" ]] && ! is_git_checkout "${flash_attention_dir}"; then
  die "${flash_attention_dir} exists but is not a git checkout"
fi

if [[ ! -e "${flash_attention_dir}" ]]; then
  mkdir -p "$(dirname "${flash_attention_dir}")"
  git clone "${flash_attention_repo}" "${flash_attention_dir}"
fi

already_prepared=0
if [[ "${force_reset}" == "1" ]]; then
  git -C "${flash_attention_dir}" reset --hard
else
  current_commit="$(git -C "${flash_attention_dir}" rev-parse HEAD)"
  if tree_is_clean "${flash_attention_dir}"; then
    :
  elif [[ "${current_commit}" == "${flash_attention_base_commit}" ]] &&
       patches_are_applied "${flash_attention_dir}"; then
    already_prepared=1
  else
    die "${flash_attention_dir} has local changes; set FLASH_ATTENTION_FORCE=1 to reset them"
  fi
fi

ensure_commit "${flash_attention_dir}" "${flash_attention_base_commit}"
if [[ "${already_prepared}" != "1" ]]; then
  git -C "${flash_attention_dir}" checkout --detach "${flash_attention_base_commit}"
fi

git -C "${flash_attention_dir}" submodule update --init csrc/cutlass
ensure_commit "${flash_attention_dir}/csrc/cutlass" "${cutlass_commit}"
git -C "${flash_attention_dir}/csrc/cutlass" checkout --detach "${cutlass_commit}"

for patch in "${patches[@]}"; do
  [[ -f "${patch}" ]] || die "missing patch ${patch}"
  if git -C "${flash_attention_dir}" apply --check "${patch}" 2>/dev/null; then
    git -C "${flash_attention_dir}" apply "${patch}"
  elif git -C "${flash_attention_dir}" apply --reverse --check "${patch}" 2>/dev/null; then
    echo "patch already applied: ${patch}"
  else
    die "patch does not apply cleanly: ${patch}"
  fi
done

echo "Prepared ${flash_attention_dir}"
git -C "${flash_attention_dir}" status --short
