#!/usr/bin/env python3
"""Applies the site's theme preset to the generated Reference Architecture pages.

  python3 .github/scripts/skin_reference_architecture.py          apply (safe to repeat)
  python3 .github/scripts/skin_reference_architecture.py --check  exit 1 if the skin is missing or stale

docs/reference-architecture/ is a verbatim copy of a static site generated in another repository.
All of its ~1,600 pages share two files, assets/style.css and assets/theme.js, so the whole
section can be restyled without touching a single page. This script appends one generated block
to each of those files, built from the active preset (docs/_config.yml `theme_preset`) in
docs/_data/themes/. The block sits between ND-SKIN BEGIN / ND-SKIN END markers and is replaced,
never duplicated, on every run.

It also (1) makes the section share the light/dark choice with the rest of the site, by using the
same localStorage key, (2) adds a "Study guide" link back to the main site, and (3) adds the same
GSA footer note as the main site.

Run it after every refresh of docs/reference-architecture/ from the generator, and after changing
the preset. Standard library only.
"""
import re, sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from check_theme import read_tokens  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
DOCS = ROOT / "docs"
ASSETS = DOCS / "reference-architecture" / "assets"
BEGIN, END = "ND-SKIN BEGIN", "ND-SKIN END"


def rgba(hexcolor, alpha):
    h = hexcolor.lstrip("#")
    return "rgba(%d, %d, %d, %s)" % (int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16), alpha)


def variables(t, dark):
    """The generator's variable names, filled from the preset's tokens."""
    return (
        f"--page: {t['surface']}; --surface: {t['surface-raised']}; --ink: {t['ink']}; "
        f"--ink-2: {t['ink-soft']}; --muted: {t['muted']}; --hair: {t['hairline']}; "
        f"--axis: {t['axis']}; --accent: {t['link']}; --accent-ink: {t['link']}; "
        f"--edge: {rgba(t['ink-soft'], 0.30 if dark else 0.35)}; --rule: {t['rule']};"
    )


def build_css(preset, light, dark):
    shared = (
        f"--radius: {light['radius']}; --font: {light['font-sans']}; --mono: {light['font-mono']};"
    )
    css = f"""/* {BEGIN} — generated from theme preset "{preset}" by .github/scripts/skin_reference_architecture.py.
   Do not edit by hand: change docs/_data/themes/{preset}.yml and re-run the script. */
:root {{ {variables(light, False)} {shared} }}
@media (prefers-color-scheme: dark) {{
  :root:where(:not([data-theme="light"])) {{ {variables(dark, True)} }}
}}
:root[data-theme="dark"] {{ {variables(dark, True)} }}

body {{
  background: var(--page); color: var(--ink); font-family: var(--font);
  font-weight: {light['weight-body']}; letter-spacing: {light['tracking-body']};
  -webkit-font-smoothing: antialiased;
}}
h1, h2, h3 {{ font-weight: {light['weight-strong']}; letter-spacing: {light['tracking-heading']}; line-height: {light['leading-heading']}; }}
h1 {{ font-size: {light['h1-size']}; }}
:focus-visible {{ outline: 2px solid var(--ink); outline-offset: 2px; }}

/* header */
.site-header {{ background: var(--page); border-bottom-color: var(--hair); align-items: center; }}
.brand {{ font-weight: {light['weight-strong']}; letter-spacing: {light['tracking-heading']}; }}
.site-header nav a {{ border-radius: var(--radius); color: var(--ink-2); }}
.site-header nav a[aria-current="page"], .site-header nav a:hover {{
  background: transparent; color: var(--ink); box-shadow: inset 0 -2px 0 var(--ink);
}}
.nd-back {{ color: var(--ink-2); font-size: 0.9rem; white-space: nowrap; padding-right: 0.9rem; border-right: 1px solid var(--hair); }}
.nd-back:hover {{ color: var(--ink); }}
.theme-toggle {{
  width: 2rem; height: 2rem; padding: 0; display: inline-flex; align-items: center; justify-content: center;
  border: 1px solid var(--ink); border-radius: var(--radius); background: transparent; color: var(--ink);
}}
.theme-toggle:hover {{ background: var(--ink); color: var(--page); }}

/* shapes: square, hairline, flat */
.badge, .tag, .chip, .src {{ border-radius: var(--radius); }}
.criterion {{ border-left-color: var(--ink); border-radius: 0; }}
.search {{ border-radius: var(--radius); border-color: var(--axis); background: var(--page); }}
.search:focus {{ outline-color: var(--ink); }}
.legend-item, .button {{ border-radius: var(--radius); }}
.button {{ border-color: var(--ink); background: transparent; font-weight: {light['weight-strong']}; }}
.button:hover {{ background: var(--ink); color: var(--page); }}
.tile b {{ font-weight: {light['weight-strong']}; letter-spacing: {light['tracking-heading']}; }}
.tile, .card, .rel, .vp, .profile, .ego-wrap {{ border-color: var(--hair); }}

/* tables: hairlines only */
table {{ background: transparent; border: 0; border-radius: 0; }}
th {{ background: transparent; color: var(--muted); font-weight: {light['weight-strong']}; border-bottom: 1px solid var(--rule); }}

/* note added by theme.js */
.nd-note {{ margin: 0.6rem 0 0; }}
.nd-note a {{ color: var(--ink); text-decoration: underline; }}
/* {END} */
"""
    return css


JS_BLOCK = f"""/* {BEGIN}: generated by .github/scripts/skin_reference_architecture.py; do not edit by hand. */
document.addEventListener("DOMContentLoaded", function () {{
  var i = location.pathname.indexOf("/reference-architecture/");
  var base = i >= 0 ? location.pathname.slice(0, i + 1) : "/";
  var brand = document.querySelector(".site-header .brand");
  if (brand) {{
    var back = document.createElement("a");
    back.className = "nd-back";
    back.href = base + "architecture/";
    back.textContent = "\\u2190 Study guide";
    brand.parentNode.insertBefore(back, brand);
  }}
  var foot = document.querySelector(".site-footer");
  if (foot) {{
    var note = document.createElement("p");
    note.className = "nd-note";
    note.append("Part of the GSA GitHub organization: an open source project of the U.S. General Services Administration, run under the ");
    var a = document.createElement("a");
    a.href = "https://open.gsa.gov/oss-policy/";
    a.textContent = "GSA Open Source Policy";
    note.append(a, ".");
    foot.append(note);
  }}
}});
/* {END} */
"""


def replace_block(text, block, kind):
    pat = re.compile(r"\n?/\* " + BEGIN + r".*?" + END + r" \*/\n?", re.S)
    text = pat.sub("\n", text).rstrip("\n") + "\n"
    return text + "\n" + block


def main():
    check = "--check" in sys.argv
    cfg = (DOCS / "_config.yml").read_text(encoding="utf-8")
    preset = re.search(r"^theme_preset:\s*(\S+)", cfg, re.M).group(1)
    path = DOCS / "_data" / "themes" / f"{preset}.yml"
    light, dark_overrides = read_tokens(path), read_tokens(path, "dark")
    if not dark_overrides:
        sys.exit(f"Preset {preset} has no dark: block; the Reference Architecture pages need one.")
    dark = {**light, **dark_overrides}

    css_path, js_path = ASSETS / "style.css", ASSETS / "theme.js"
    css_old, js_old = css_path.read_text(encoding="utf-8"), js_path.read_text(encoding="utf-8")
    css_new = replace_block(css_old, build_css(preset, light, dark), "css")
    js_new = js_old.replace('"ara-theme"', '"nd-theme"')   # share the light/dark choice with the main site
    js_new = replace_block(js_new, JS_BLOCK, "js")

    if check:
        stale = [p.name for p, a, b in ((css_path, css_old, css_new), (js_path, js_old, js_new)) if a != b]
        if stale:
            print("Reference Architecture skin is missing or out of date:", ", ".join(stale))
            print("Run: python3 .github/scripts/skin_reference_architecture.py")
            return 1
        print(f"Reference Architecture skin is current (preset {preset}).")
        return 0
    css_path.write_text(css_new, encoding="utf-8")
    js_path.write_text(js_new, encoding="utf-8")
    print(f"Applied preset {preset} to {css_path.relative_to(ROOT)} and {js_path.relative_to(ROOT)}.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
