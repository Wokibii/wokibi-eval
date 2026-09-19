#!/usr/bin/env bash
# use-local-wokibi.sh — link sibling Wokibi libraries into this repo's Poetry venv (editable).
# Keep in sync with wokibi-pipelines/scripts/use-local-wokibi.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONSUMER_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
WOKIBI_ROOT="${WOKIBI_ROOT:-$(dirname "$CONSUMER_ROOT")}"

PKG_ORDER=(cliid data ai)

pkg_dir() {
  case "$1" in
    cliid) echo "${WOKIBI_ROOT}/cliid" ;;
    data) echo "${WOKIBI_ROOT}/wokibi-data" ;;
    ai) echo "${WOKIBI_ROOT}/wokibi-ai" ;;
    *)
      echo "error: unknown package alias '$1'" >&2
      return 1
      ;;
  esac
}

usage() {
  cat <<'EOF'
Usage: ./scripts/use-local-wokibi.sh <command> [args...]

Commands:
  link [all|cliid|data|ai ...]   pip install -e sibling repos (default: all)
  status                         show import paths (local vs site-packages)
  reset                          poetry install --sync (restore lock / TestPyPI)
  help|--help|-h                 this message

Environment:
  WOKIBI_ROOT   Parent folder of sibling repos (default: parent of this consumer repo)
EOF
}

require_poetry_project() {
  if [[ ! -f "${CONSUMER_ROOT}/pyproject.toml" ]]; then
    echo "error: no pyproject.toml in ${CONSUMER_ROOT}" >&2
    exit 1
  fi
  if ! command -v poetry >/dev/null 2>&1; then
    echo "error: poetry not found in PATH" >&2
    exit 1
  fi
}

validate_sibling_dirs() {
  local alias dir
  local missing=0
  for alias in "${PKG_ORDER[@]}"; do
    dir="$(pkg_dir "$alias")"
    if [[ ! -d "$dir" ]]; then
      echo "  missing: ${dir}" >&2
      missing=1
    fi
  done
  if [[ "$missing" -eq 1 ]]; then
    echo "error: expected sibling repos under WOKIBI_ROOT=${WOKIBI_ROOT}" >&2
    exit 1
  fi
}

want_pkg() {
  local alias="$1"
  shift
  if (($# == 0)); then return 0; fi
  local arg
  for arg in "$@"; do
    if [[ "$arg" == "$alias" ]]; then return 0; fi
  done
  return 1
}

cmd_link() {
  validate_sibling_dirs
  local selection=()
  if (($# == 0)) || [[ "${1:-}" == "all" ]]; then
    selection=("${PKG_ORDER[@]}")
  else
    local arg
    for arg in "$@"; do
      case "$arg" in
        all) selection=("${PKG_ORDER[@]}") ;;
        cliid | data | ai) selection+=("$arg") ;;
        *)
          echo "error: unknown package alias '${arg}'" >&2
          exit 1
          ;;
      esac
    done
  fi
  cd "$CONSUMER_ROOT"
  local alias dir
  for alias in "${PKG_ORDER[@]}"; do
    if want_pkg "$alias" "${selection[@]}"; then
      dir="$(pkg_dir "$alias")"
      echo "==> editable install: ${alias} (${dir})"
      poetry run pip install -e "${dir}"
    fi
  done
}

cmd_status() {
  cd "$CONSUMER_ROOT"
  export WOKIBI_ROOT
  poetry run python - <<'PY'
import importlib
import os

wokibi_root = os.path.realpath(os.environ["WOKIBI_ROOT"])
sibling_roots = [
    os.path.join(wokibi_root, "cliid"),
    os.path.join(wokibi_root, "wokibi-data"),
    os.path.join(wokibi_root, "wokibi-ai"),
]
sibling_roots = [os.path.realpath(p) for p in sibling_roots]

def is_sibling_source(path: str) -> bool:
    for root in sibling_roots:
        if path == root or path.startswith(root + os.sep):
            return True
    return False

for label, mod_name in [("cliid", "cliid"), ("data", "wokibi_data"), ("ai", "wokibi_ai")]:
    try:
        mod = importlib.import_module(mod_name)
        path = os.path.realpath(getattr(mod, "__file__", "") or "")
    except Exception as exc:
        print(f"{label:5}  {mod_name}: NOT IMPORTABLE ({exc})")
        continue
    kind = "local" if is_sibling_source(path) else ("pypi/lock" if "site-packages" in path else "other")
    print(f"{label:5}  {mod_name}: [{kind}]")
    print(f"       {path}")
PY
}

cmd_reset() {
  cd "$CONSUMER_ROOT"
  poetry install --sync
}

main() {
  require_poetry_project
  local cmd="${1:-help}"
  shift || true
  case "$cmd" in
    link) cmd_link "$@" ;;
    status) cmd_status ;;
    reset) cmd_reset ;;
    help | --help | -h) usage ;;
    *)
      echo "error: unknown command '${cmd}'" >&2
      usage >&2
      exit 1
      ;;
  esac
}

main "$@"
