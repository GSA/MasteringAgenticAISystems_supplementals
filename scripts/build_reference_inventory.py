#!/usr/bin/env python3
"""Write a deterministic CSV inventory for the References directory."""

from __future__ import annotations

import csv
import hashlib
from pathlib import Path
import sys


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    output = Path(sys.argv[1]) if len(sys.argv) > 1 else root / 'References/reference_inventory.csv'
    if not output.is_absolute():
        output = root / output

    reference_root = root / 'References'
    rows = []
    for path in sorted(p for p in reference_root.rglob('*') if p.is_file() and p.resolve() != output.resolve()):
        rel = path.relative_to(root).as_posix()
        chapter = path.relative_to(reference_root).parts[0]
        if not chapter.startswith('Chapter '):
            chapter = '(top level)'
        rows.append((rel, chapter, path.suffix.lower(), path.stat().st_size, sha256(path)))

    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow(['path', 'chapter_or_group', 'extension', 'size_bytes', 'sha256'])
        writer.writerows(rows)
    print(f'Wrote {len(rows)} records to {output}')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
