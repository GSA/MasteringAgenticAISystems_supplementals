#!/usr/bin/env python3
"""Checks the site's theme presets. Run from anywhere:  python3 .github/scripts/check_theme.py

  1. Every preset in docs/_data/themes/ defines the same tokens as the reference preset, and
     any `dark:` block overrides only known tokens.
  2. Text and control colours meet WCAG AA contrast (4.5:1; 3:1 for the focus ring), in the
     light tokens and, where a preset has one, in the dark theme.
  3. The stylesheets in docs/_sass/custom/ contain no literal colours (use var(--nd-*)).
  4. `theme_preset` in docs/_config.yml names a preset that exists.
  5. The Reference Architecture pages carry the current preset (skin_reference_architecture.py --check).

Uses only the Python standard library. Exits 1 if anything fails.
"""
import re, subprocess, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
REFERENCE = "ndstudio"

# (foreground token, background token, minimum ratio, what it is)
PAIRS = [
    ("ink", "surface", 4.5, "main text"),
    ("ink", "surface-raised", 4.5, "text on cards and code"),
    ("ink-soft", "surface", 4.5, "navigation links"),
    ("ink-soft", "surface-raised", 4.5, "card descriptions"),
    ("muted", "surface", 4.5, "captions and table headers"),
    ("muted", "surface-raised", 4.5, "captions on cards"),
    ("link", "surface", 4.5, "links"),
    ("link", "surface-raised", 4.5, "links on cards"),
    ("link-hover", "surface", 4.5, "hovered links"),
    ("surface", "ink", 4.5, "button and card hover text"),
    ("surface", "link", 4.5, "primary button hover text"),
    ("danger", "surface", 4.5, "warnings"),
    ("focus", "surface", 3.0, "keyboard focus ring"),
]


def read_tokens(path, block="tokens"):
    """Minimal reader for a flat top-level block such as `tokens:` or `dark:` (no PyYAML needed)."""
    tokens, inside = {}, False
    for line in path.read_text(encoding="utf-8").splitlines():
        if re.match(rf"^{block}:\s*$", line):
            inside = True
            continue
        if inside:
            m = re.match(r"^  ([a-z0-9-]+):\s*(.+?)\s*(?:\s#.*)?$", line)
            if m:
                v = m.group(2).strip()
                if v[:1] in "\"'" and v[-1:] == v[:1]:
                    v = v[1:-1]
                tokens[m.group(1)] = v
            elif line.strip() and not line.lstrip().startswith("#"):
                inside = False
    return tokens


def lum(hexcolor):
    h = hexcolor.lstrip("#")
    if len(h) == 3:
        h = "".join(c * 2 for c in h)
    r, g, b = (int(h[i:i + 2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)


def ratio(a, b):
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def main():
    problems = []
    files = sorted((DOCS / "_data/themes").glob("*.yml"))
    presets = {p.stem: read_tokens(p) for p in files}
    darks = {p.stem: read_tokens(p, "dark") for p in files}
    if REFERENCE not in presets:
        sys.exit(f"Reference preset {REFERENCE}.yml is missing.")
    ref = set(presets[REFERENCE])

    for name, tok in presets.items():
        for missing in sorted(ref - set(tok)):
            problems.append(f"[{name}] missing token: {missing}")
        for extra in sorted(set(tok) - ref):
            problems.append(f"[{name}] unknown token (not in {REFERENCE}): {extra}")
        for extra in sorted(set(darks[name]) - ref):
            problems.append(f"[{name}] unknown token in dark block: {extra}")
        themes = [("light", tok)]
        if darks[name]:
            themes.append(("dark", {**tok, **darks[name]}))
        for mode, tk in themes:
            for fg, bg, need, what in PAIRS:
                if fg in tk and bg in tk:
                    label = f"{name}/{mode}"
                    try:
                        r = ratio(tk[fg], tk[bg])
                    except ValueError:
                        problems.append(f"[{label}] {fg}/{bg} must be #rrggbb hex colours")
                        continue
                    status = "ok " if r >= need else "LOW"
                    print(f"  {label:14s} {status} {r:5.2f}:1  {fg} on {bg} ({what}, needs {need})")
                    if r < need:
                        problems.append(f"[{label}] {fg} on {bg} is {r:.2f}:1, needs {need}:1 ({what})")
        if REFERENCE == name and not darks[name]:
            problems.append(f"[{name}] the reference preset must define a dark: block")

    for scss in sorted((DOCS / "_sass/custom").rglob("*.scss")):
        text = re.sub(r"//.*|/\*.*?\*/", "", scss.read_text(encoding="utf-8"), flags=re.S)
        for m in re.finditer(r"#[0-9a-fA-F]{3,8}\b|\brgba?\(|\bhsla?\(", text):
            problems.append(f"{scss.relative_to(ROOT)}: literal colour {m.group(0)!r}; use a var(--nd-*) token")

    cfg = (DOCS / "_config.yml").read_text(encoding="utf-8")
    m = re.search(r"^theme_preset:\s*(\S+)", cfg, re.M)
    if not m or m.group(1) not in presets:
        problems.append(f"_config.yml theme_preset {m.group(1) if m else '(missing)'} has no file in docs/_data/themes/")

    skin = Path(__file__).resolve().parent / "skin_reference_architecture.py"
    if skin.exists():
        run = subprocess.run([sys.executable, str(skin), "--check"], capture_output=True, text=True)
        if run.returncode != 0:
            problems.append("Reference Architecture skin is out of date; run python3 .github/scripts/skin_reference_architecture.py")
        else:
            print("  " + run.stdout.strip())

    if problems:
        print("\nTheme check FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nTheme check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
