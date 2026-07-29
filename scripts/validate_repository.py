#!/usr/bin/env python3
"""Validate first-party repository structure without third-party dependencies."""

from __future__ import annotations

import ast
from collections import Counter
import csv
import html
import json
import os
from pathlib import Path
import re
import sys
import textwrap
from urllib.parse import unquote, urlsplit

ROOT = Path(__file__).resolve().parents[1]
EXCLUDED_LINK_TREES = {'References'}
EXCLUDED_PRIVATE_PATH_TREES = {'References'}
REQUIRED_FILES = {
    'README.md', 'LICENSE.md', 'CONTRIBUTING.md', 'CODE_OF_CONDUCT.md',
    'SECURITY.md', 'THIRD_PARTY_NOTICES.md', '.codeinventory.yml',
    '.gitattributes', '.github/workflows/validate.yml',
    'Study_Plan.md', 'code_examples/README.md', 'code_examples/COMPATIBILITY.md',
    'code_examples/fragments.txt', 'code_examples/scaffolds.txt',
}
LINK_RE = re.compile(r'!?\[[^\]]*\]\(([^)]+)\)')
HEADING_RE = re.compile(r'^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$')
HTML_ANCHOR_RE = re.compile(
    r'<a\s+(?:[^>]*?\s)?(?:id|name)=[\"\']([^\"\']+)[\"\'][^>]*>',
    re.IGNORECASE,
)
PRIVATE_PATH_RE = re.compile(r'(?<![A-Za-z0-9_])/Users/[A-Za-z0-9._~-]+/')
SCAFFOLD_MARKER_RE = re.compile(r'\b(?:TODO|FIXME|TBD)\b')
HARD_CODED_PLACEHOLDER_CREDENTIAL_RE = re.compile(
    r'(?i)(?:api[_-]?key|token|password)\s*[=:]\s*["\']'
    r'(?:your|replace|changeme|example|test|dummy|secret)[^"\']*["\']'
)
CREDENTIAL_PATTERNS = {
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA )?PRIVATE KEY-----'),
    'AWS access key': re.compile(r'\b(?:AKIA|ASIA)[A-Z0-9]{16}\b'),
    'GitHub token': re.compile(r'\bgh[pousr]_[A-Za-z0-9]{30,255}\b'),
    'OpenAI key': re.compile(r'\bsk-(?:proj-)?[A-Za-z0-9_-]{20,}\b'),
    'Google API key': re.compile(r'\bAIza[0-9A-Za-z_-]{35}\b'),
    'NVIDIA API key': re.compile(r'\bnvapi-[A-Za-z0-9_-]{24,}\b'),
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def repository_paths(pattern: str = '*'):
    """Yield repository paths while excluding Git's private metadata tree."""
    for path in ROOT.rglob(pattern):
        if '.git' not in path.relative_to(ROOT).parts:
            yield path


def load_path_manifest(name: str) -> set[str]:
    manifest = ROOT / 'code_examples' / name
    return {
        line.strip() for line in manifest.read_text(encoding='utf-8').splitlines()
        if line.strip() and not line.lstrip().startswith('#')
    }


def load_fragments() -> set[str]:
    return load_path_manifest('fragments.txt')


def load_scaffolds() -> set[str]:
    return load_path_manifest('scaffolds.txt')


def check_required(errors: list[str]) -> None:
    for item in sorted(REQUIRED_FILES):
        if not (ROOT / item).is_file():
            errors.append(f'missing required file: {item}')
    codeinventory = (ROOT / '.codeinventory.yml').read_text(encoding='utf-8')
    for placeholder in ('[PROJECT]', '[PROJECT DESCRIPTION]'):
        if placeholder in codeinventory:
            errors.append(f'.codeinventory.yml still contains {placeholder}')


def check_python(errors: list[str], warnings: list[str]) -> None:
    del warnings  # Reserved for future non-fatal validation categories.
    fragments = load_fragments()
    scaffolds = load_scaffolds()
    for label, entries in (('fragment', fragments), ('scaffold', scaffolds)):
        for listed in sorted(entries):
            if not (ROOT / listed).is_file():
                errors.append(f'{label} manifest points to a missing file: {listed}')

    for path in sorted(repository_paths('*.py')):
        path_rel = rel(path)
        if path_rel.startswith('References/'):
            continue
        try:
            ast.parse(path.read_text(encoding='utf-8'), filename=path_rel)
        except (SyntaxError, UnicodeDecodeError) as exc:
            errors.append(f'Python parse failure: {path_rel}: {exc}')

    scaffold_suffixes = {'.py', '.js', '.sh', '.yaml', '.yml'}
    marker_files: set[str] = set()
    code_root = ROOT / 'code_examples'
    for path in sorted(p for p in code_root.rglob('*') if p.is_file() and p.suffix.lower() in scaffold_suffixes):
        path_rel = rel(path)
        text = path.read_text(encoding='utf-8', errors='replace')
        if SCAFFOLD_MARKER_RE.search(text):
            marker_files.add(path_rel)
            if path_rel not in scaffolds:
                errors.append(f'unregistered exercise scaffold marker: {path_rel}')
    for listed in sorted(scaffolds - marker_files):
        errors.append(f'scaffold manifest entry has no TODO/FIXME/TBD marker: {listed}')


def check_json_csv(errors: list[str]) -> None:
    for path in sorted(repository_paths('*.json')):
        if rel(path).startswith('References/'):
            continue
        try:
            json.loads(path.read_text(encoding='utf-8'))
        except Exception as exc:
            errors.append(f'JSON parse failure: {rel(path)}: {exc}')

    allowed_ratings = {'H', 'M', 'L', 'N'}
    for path in sorted(repository_paths('*.csv')):
        if rel(path).startswith('References/'):
            continue
        try:
            with path.open(encoding='utf-8-sig', newline='') as handle:
                rows = list(csv.reader(handle))
        except Exception as exc:
            errors.append(f'CSV read failure: {rel(path)}: {exc}')
            continue
        if not rows:
            errors.append(f'empty CSV: {rel(path)}')
            continue
        width = len(rows[0])
        for number, row in enumerate(rows[1:], 2):
            if len(row) != width:
                errors.append(f'non-rectangular CSV: {rel(path)} row {number}: {len(row)} != {width}')
        if path.parent.name == 'cert_mapping':
            for number, row in enumerate(rows[1:], 2):
                invalid = sorted({cell.strip() for cell in row[1:] if cell.strip() not in allowed_ratings})
                if invalid:
                    errors.append(f'invalid certification rating in {rel(path)} row {number}: {invalid}')


def split_markdown_destination(raw: str) -> str:
    raw = raw.strip()
    if raw.startswith('<') and '>' in raw:
        return raw[1:raw.index('>')]
    titled = re.match(r'^(.*?)(?:\s+["\'][^"\']*["\'])$', raw)
    return titled.group(1).strip() if titled else raw


def github_slug(title: str) -> str:
    """Approximate GitHub's heading slug rules for the repository's Markdown."""
    title = re.sub(r'<[^>]+>', '', title)
    title = html.unescape(title).strip().lower()
    title = re.sub(r'[^\w\- ]', '', title, flags=re.UNICODE)
    return title.replace(' ', '-')


def markdown_anchors(path: Path) -> set[str]:
    text = path.read_text(encoding='utf-8', errors='replace')
    anchors = set(HTML_ANCHOR_RE.findall(text))
    counts: Counter[str] = Counter()
    in_fence = False
    fence_marker = ''
    for line in text.splitlines():
        fence = re.match(r'^\s*(`{3,}|~{3,})', line)
        if fence:
            marker = fence.group(1)[0]
            if not in_fence:
                in_fence = True
                fence_marker = marker
            elif marker == fence_marker:
                in_fence = False
                fence_marker = ''
            continue
        if in_fence:
            continue
        heading = HEADING_RE.match(line)
        if not heading:
            continue
        base = github_slug(heading.group(2))
        number = counts[base]
        counts[base] += 1
        anchors.add(base if number == 0 else f'{base}-{number}')
    return anchors


def resolve_markdown_target(source: Path, raw: str) -> tuple[Path | None, str | None, str]:
    destination = split_markdown_destination(raw).strip().strip('<>')
    if not destination:
        return None, None, destination
    parsed = urlsplit(destination)
    if parsed.scheme or parsed.netloc or destination.startswith('//'):
        return None, None, destination
    path_text = unquote(parsed.path)
    fragment = unquote(parsed.fragment) or None
    candidate = source.resolve() if not path_text else (source.parent / path_text).resolve()
    return candidate, fragment, destination


def check_markdown_links(errors: list[str]) -> None:
    root = ROOT.resolve()
    anchor_cache: dict[Path, set[str]] = {}
    for path in sorted(repository_paths('*.md')):
        path_rel = rel(path)
        if path_rel.split('/', 1)[0] in EXCLUDED_LINK_TREES:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for match in LINK_RE.finditer(text):
            target, fragment, destination = resolve_markdown_target(path, match.group(1))
            if target is None:
                continue
            line = text.count('\n', 0, match.start()) + 1
            try:
                target.relative_to(root)
            except ValueError:
                errors.append(f'local Markdown link escapes repository: {path_rel}:{line} -> {destination}')
                continue
            if not target.exists():
                errors.append(f'broken local Markdown link: {path_rel}:{line} -> {destination}')
                continue
            if not fragment:
                continue
            anchor_file = target / 'README.md' if target.is_dir() else target
            if anchor_file.suffix.lower() != '.md' or not anchor_file.is_file():
                continue
            anchors = anchor_cache.setdefault(anchor_file, markdown_anchors(anchor_file))
            if fragment.lower() not in anchors and github_slug(fragment) not in anchors:
                errors.append(f'broken Markdown anchor: {path_rel}:{line} -> {destination}')



def check_study_plan_structure(errors: list[str]) -> None:
    path = ROOT / 'Study_Plan.md'
    lines = path.read_text(encoding='utf-8').splitlines()
    try:
        toc_start = lines.index('## Table of Contents') + 1
        toc_end = next(index for index in range(toc_start, len(lines)) if lines[index] == '---')
    except (ValueError, StopIteration):
        errors.append('Study_Plan.md lacks a bounded Table of Contents section')
        return

    toc: dict[int, dict[str, object]] = {}
    current_part: int | None = None
    part_re = re.compile(r'^### Part (\d+): .+ \((\d+) chapters\)$')
    item_re = re.compile(r'^- \[([^:]+): (.+?)\]\(')
    for line in lines[toc_start:toc_end]:
        part_match = part_re.match(line)
        if part_match:
            current_part = int(part_match.group(1))
            toc[current_part] = {'declared': int(part_match.group(2)), 'entries': []}
            continue
        item_match = item_re.match(line)
        if item_match and current_part is not None:
            entries = toc[current_part]['entries']
            assert isinstance(entries, list)
            entries.append((item_match.group(1), item_match.group(2)))

    actual: dict[int, list[tuple[str, str]]] = {}
    heading_re = re.compile(r'^## Part (\d+), Chapter ([^:]+): (.+)$')
    for line in lines:
        heading_match = heading_re.match(line)
        if heading_match:
            actual.setdefault(int(heading_match.group(1)), []).append(
                (heading_match.group(2), heading_match.group(3))
            )

    for part in sorted(set(toc) | set(actual)):
        if part not in toc:
            errors.append(f'Study_Plan.md TOC omits Part {part}')
            continue
        if part not in actual:
            errors.append(f'Study_Plan.md TOC lists Part {part} with no chapter headings')
            continue
        declared = toc[part]['declared']
        entries = toc[part]['entries']
        assert isinstance(declared, int) and isinstance(entries, list)
        if declared != len(entries):
            errors.append(
                f'Study_Plan.md Part {part} declares {declared} chapters but lists {len(entries)} TOC entries'
            )
        if entries != actual[part]:
            toc_ids = [entry[0] for entry in entries]
            actual_ids = [entry[0] for entry in actual[part]]
            errors.append(
                f'Study_Plan.md Part {part} TOC/headings differ: TOC {toc_ids}; headings {actual_ids}'
            )


def check_markdown_fences(errors: list[str]) -> None:
    for path in sorted(repository_paths('*.md')):
        path_rel = rel(path)
        if path_rel.startswith('References/'):
            continue
        marker: str | None = None
        length = 0
        opening_line = 0
        for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            match = re.match(r'^\s{0,3}(`{3,}|~{3,})', line)
            if not match:
                continue
            candidate = match.group(1)
            if marker is None:
                marker = candidate[0]
                length = len(candidate)
                opening_line = number
            elif candidate[0] == marker and len(candidate) >= length:
                marker = None
                length = 0
                opening_line = 0
        if marker is not None:
            errors.append(f'unclosed Markdown fence: {path_rel}:{opening_line}')


def check_markdown_code_blocks(errors: list[str]) -> None:
    supported = {'python': 'python', 'py': 'python', 'json': 'json'}
    for path in sorted(repository_paths('*.md')):
        path_rel = rel(path)
        if path_rel.startswith('References/'):
            continue
        in_fence = False
        marker = ''
        length = 0
        language = ''
        opening_line = 0
        body: list[str] = []
        for number, line in enumerate(path.read_text(encoding='utf-8', errors='replace').splitlines(), 1):
            match = re.match(r'^\s{0,3}(`{3,}|~{3,})\s*([^\s`]*)', line)
            if not in_fence:
                if match:
                    in_fence = True
                    marker = match.group(1)[0]
                    length = len(match.group(1))
                    language = match.group(2).lower()
                    opening_line = number
                    body = []
                continue
            if match and match.group(1)[0] == marker and len(match.group(1)) >= length:
                kind = supported.get(language)
                if kind:
                    code = textwrap.dedent('\n'.join(body)) + '\n'
                    try:
                        if kind == 'python':
                            ast.parse(code, filename=f'{path_rel}:{opening_line}')
                        else:
                            json.loads(code)
                    except Exception as exc:
                        errors.append(
                            f'{kind} fenced-block parse failure: {path_rel}:{opening_line}: {exc}'
                        )
                in_fence = False
                marker = ''
                length = 0
                language = ''
                opening_line = 0
                body = []
            else:
                body.append(line)

def check_private_paths(errors: list[str]) -> None:
    text_suffixes = {'.md', '.txt', '.py', '.js', '.sh', '.yaml', '.yml', '.json', '.csv', '.rst', '.html'}
    for path in sorted(p for p in repository_paths('*') if p.is_file() and p.suffix.lower() in text_suffixes):
        path_rel = rel(path)
        if path_rel.split('/', 1)[0] in EXCLUDED_PRIVATE_PATH_TREES:
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        if PRIVATE_PATH_RE.search(text):
            errors.append(f'private absolute path found: {path_rel}')


def check_known_regressions(errors: list[str]) -> None:
    forbidden = {
        'code_examples/Part_02_Chapter_2.9_sse_client_consumer_code_03_sse_client_consumer.py': 'JavaScript example has a .py suffix',
        'cert_mapping/microsoft_AI#U2011102.md': 'encoded filename artifact remains',
        'slides/Chapter1.7B_1.8_v1.0_2026_03_01 (1).pdf': 'duplicate-download filename remains',
    }
    for path, message in forbidden.items():
        if (ROOT / path).exists():
            errors.append(f'{message}: {path}')
    for path in (ROOT / 'videos').glob('*.md'):
        text = path.read_text(encoding='utf-8', errors='replace')
        if 'YOUR_URL' in text:
            errors.append(f'unresolved video URL placeholder: {rel(path)}')


def check_symlinks(errors: list[str]) -> None:
    for path in sorted(repository_paths('*')):
        if path.is_symlink() and not path.exists():
            errors.append(f'broken symlink: {rel(path)} -> {path.readlink()}')




def check_credentials(errors: list[str]) -> None:
    text_suffixes = {'.md', '.txt', '.py', '.js', '.sh', '.yaml', '.yml', '.json', '.csv', '.rst', '.html'}
    for path in sorted(p for p in repository_paths('*') if p.is_file() and p.suffix.lower() in text_suffixes):
        path_rel = rel(path)
        if path_rel.startswith('References/'):
            continue
        text = path.read_text(encoding='utf-8', errors='replace')
        for label, pattern in CREDENTIAL_PATTERNS.items():
            if pattern.search(text):
                errors.append(f'possible committed {label}: {path_rel}')
        if path_rel.startswith('code_examples/') and HARD_CODED_PLACEHOLDER_CREDENTIAL_RE.search(text):
            errors.append(f'hard-coded placeholder credential in code example: {path_rel}')

def check_repository_hygiene(errors: list[str]) -> None:
    forbidden_names = {'.DS_Store'}
    for path in sorted(repository_paths('*')):
        path_rel = rel(path)
        if path.name in forbidden_names or path.suffix == '.pyc' or '__pycache__' in path.parts:
            errors.append(f'generated local artifact committed: {path_rel}')
        if re.search(r'#U[0-9A-Fa-f]{4,6}', path.name):
            errors.append(f'encoded filename artifact remains: {path_rel}')
        if path.is_file() and path.stat().st_size > 100 * 1024 * 1024:
            errors.append(f'file exceeds GitHub 100 MiB object limit: {path_rel}')

    entrypoints = (
        ROOT / 'scripts' / 'build_reference_inventory.py',
        ROOT / 'scripts' / 'validate_repository.py',
        ROOT / 'videos' / 'validate_video_resources.py',
    )
    for path in entrypoints:
        if not path.is_file():
            continue
        first_line = path.read_text(encoding='utf-8', errors='replace').splitlines()[:1]
        if not first_line or not first_line[0].startswith('#!'):
            errors.append(f'utility script lacks a first-line shebang: {rel(path)}')
        if os.name == 'posix' and not (path.stat().st_mode & 0o111):
            errors.append(f'utility script is not executable: {rel(path)}')

    if os.name == 'posix':
        for path in sorted(repository_paths('*.onnx')):
            if path.stat().st_mode & 0o111:
                errors.append(f'binary model file is unexpectedly executable: {rel(path)}')


def check_text_hygiene(errors: list[str]) -> None:
    text_suffixes = {
        '.md', '.txt', '.py', '.js', '.sh', '.yaml', '.yml', '.json', '.csv',
        '.rst', '.html', '.toml', '.ini', '.cfg', '.conf',
    }
    special_names = {'Makefile', '.gitattributes', '.gitignore', '.codeinventory.yml', '.pre-commit-config.yaml'}
    for path in sorted(p for p in repository_paths('*') if p.is_file()):
        path_rel = rel(path)
        if path_rel.startswith('References/'):
            continue
        if path.suffix.lower() not in text_suffixes and path.name not in special_names:
            continue
        data = path.read_bytes()
        if b'\r\n' in data:
            errors.append(f'CRLF line endings in first-party text file: {path_rel}')
        if data and not data.endswith(b'\n'):
            errors.append(f'first-party text file lacks a final newline: {path_rel}')
        try:
            decoded = data.decode('utf-8')
        except UnicodeDecodeError:
            errors.append(f'first-party text file is not UTF-8: {path_rel}')
            continue
        for number, line in enumerate(decoded.splitlines(), 1):
            if line.endswith((' ', '\t')):
                errors.append(f'trailing whitespace in first-party text file: {path_rel}:{number}')

def main() -> int:
    errors: list[str] = []
    warnings: list[str] = []
    check_required(errors)
    check_python(errors, warnings)
    check_json_csv(errors)
    check_markdown_links(errors)
    check_study_plan_structure(errors)
    check_markdown_fences(errors)
    check_markdown_code_blocks(errors)
    check_private_paths(errors)
    check_known_regressions(errors)
    check_symlinks(errors)
    check_credentials(errors)
    check_repository_hygiene(errors)
    check_text_hygiene(errors)

    print(f'Validation root: {ROOT}')
    print(f'Warnings: {len(warnings)}')
    for item in warnings:
        print(f'  WARN: {item}')
    print(f'Errors: {len(errors)}')
    for item in errors:
        print(f'  ERROR: {item}')
    if errors:
        return 1
    print('Repository validation passed.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
