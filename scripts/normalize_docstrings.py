#!/usr/bin/env python3
"""Normalize Google docstrings: drop Args type parens and bare Returns: None."""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGETS = [
    ROOT / "src" / "mlkit",
    ROOT / "tests",
    ROOT / "example",
]

RETURN_NONE = re.compile(r"\n\s*Returns:\s*\n\s*None\s*\n", re.MULTILINE)
ARG_TYPED = re.compile(r"^(\s+)(\w+)\s+\([^)]+\):\s*", re.MULTILINE)
RAISES_TICK = re.compile(r"^(\s+)`([^`]+)`: ", re.MULTILINE)
RETURNS_TYPED = re.compile(
    r"^(\s+)(Returns:\s*\n)(\s+)`[^`]+`:\s*",
    re.MULTILINE,
)


def normalize_text(text: str) -> str:
    text = RETURN_NONE.sub("\n", text)
    text = ARG_TYPED.sub(r"\1\2: ", text)
    text = RAISES_TICK.sub(r"\1\2: ", text)
    text = RETURNS_TYPED.sub(r"\1\2\3", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text


def normalize_file(path: Path) -> bool:
    source = path.read_text(encoding="utf-8")
    parts: list[str] = []
    i = 0
    changed = False
    while i < len(source):
        start = source.find('"""', i)
        if start == -1:
            parts.append(source[i:])
            break
        parts.append(source[i:start])
        end = source.find('"""', start + 3)
        if end == -1:
            parts.append(source[start:])
            break
        inner = source[start + 3 : end]
        new_inner = normalize_text(inner)
        if new_inner != inner:
            changed = True
        parts.append('"""' + new_inner + '"""')
        i = end + 3
    if not changed:
        return False
    path.write_text("".join(parts), encoding="utf-8")
    return True


def main() -> int:
    roots = [Path(p) for p in sys.argv[1:]] if len(sys.argv) > 1 else TARGETS
    count = 0
    for root in roots:
        files = [root] if root.is_file() and root.suffix == ".py" else sorted(root.rglob("*.py"))
        for path in files:
            if normalize_file(path):
                count += 1
                print(path.relative_to(ROOT))
    print(f"updated {count} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())