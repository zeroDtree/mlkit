#!/usr/bin/env bash

# @help-begin
# Auto-fix lint style and run type checks from the project root.
#
# Usage:
#   bash shell_script/check-fix.sh
#
# Runs ruff with --fix (src/mlkit, tests, example per ruff.toml), format, verify, then ty.
#
# Env: MC_ENABLE_PROXY — set to 1 before this script to enable the HTTP proxy.
# @help-end

# @help-options-begin
#   -h, --help              show help
# @help-options-end

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

usage() {
	awk '/^# @help-begin$/{f=1; next} /^# @help-end$/{f=0} f' "$0"
	printf '%s\n' '#' 'Options:' '#'
	awk '/^# @help-options-begin$/{f=1; next} /^# @help-options-end$/{f=0} f' "$0"
	exit 0
}

case "${1:-}" in
	-h|--help) usage ;;
	"") ;;
	*)
		echo "Error: unknown argument: $1" >&2
		exit 1
		;;
esac

source "$SCRIPT_DIR/prepare.sh"
cd "$PROJECT_ROOT" || exit 1

echo "==> uv run ruff check --fix"
uv run ruff check --fix
echo "==> uv run ruff format"
uv run ruff format
echo "==> uv run ruff check"
uv run ruff check
echo "==> uv run ty check"
uv run ty check