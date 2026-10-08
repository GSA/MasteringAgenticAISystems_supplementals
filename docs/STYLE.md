# Styling the site

The site's look comes from one small file of values, called a **preset**. You can change
colours, fonts, corner shape and heading sizes without writing any CSS.

The default preset, `ndstudio`, is a flat, high-contrast style on a white page, with a dark theme, inspired by
[ndstudio.gov](https://ndstudio.gov/). It borrows the design language only (colour, spacing,
type scale). It uses no logos, images or font files from that site.

## Re-skin in five minutes

1. Copy `docs/_data/themes/ndstudio.yml` to `docs/_data/themes/<yourname>.yml`.
2. Edit the values under `tokens:` (the light theme) and, for colours, under `dark:` (the dark theme). Colours must be six-digit hex such as `"#1a1a1a"`.
3. In `docs/_config.yml`, set `theme_preset: <yourname>`.
4. Run `python3 .github/scripts/check_theme.py`. It fails if any text colour is hard to read.
5. Commit. GitHub Pages rebuilds the site.

Two presets ship with the repository: `ndstudio` (the default) and `classic` (an approximation
of the stock Just the Docs look). Set `theme_preset: classic` to go back.

## Light and dark

Every page has a small sun/moon button at the top of the sidebar. It switches between the light and
dark theme and remembers the choice in that browser. Until someone clicks it, the site follows the
device's own setting. If the browser blocks storage, the button still works for the current page.

The `dark:` block in a preset lists only the colour tokens that change; everything else (fonts,
sizes, corners) is shared. A preset with no `dark:` block has no dark theme and no button.
`check_theme.py` tests contrast for both themes.

## What each token controls

| Token | Controls |
|---|---|
| `surface` | Page and sidebar background (white in the default preset) |
| `surface-raised` | Cards, code blocks, search box, table row hover |
| `ink` | Headings and main text |
| `ink-soft` | Navigation links, secondary text |
| `muted` | Captions, table headers, breadcrumbs |
| `hairline` | Thin dividers |
| `axis` | Chart axes, input borders, lines in the knowledge graph |
| `rule` | Strong dividers (above sections, under table headers) |
| `link`, `link-hover` | Links; primary button on hover |
| `focus` | Keyboard focus outline |
| `danger` | Warnings |
| `radius` | Corner roundness of buttons, cards, code and inputs (`0` is square) |
| `font-sans`, `font-mono` | Font stacks for text and code |
| `tracking-body`, `tracking-heading` | Letter spacing |
| `weight-body`, `weight-strong` | Font weights for text and headings |
| `leading-heading` | Heading line height |
| `hero-size`, `h1-size`, `h2-size`, `h3-size`, `site-title-size` | Type sizes |
| `seal-size` | Width of the emblem above the site title |
| `viz-blue`, `viz-orange`, `viz-neutral` | Chart colors: main series, second series or the low end of a scale, and the middle of a scale or context |
| `viz-ramp-1` to `viz-ramp-4` | Chart colors for ordered steps, lowest to highest |

## Fonts

The ndstudio.gov site uses PP Neue Montreal, which is a licensed commercial font. This
repository does not include it. The `ndstudio` preset lists it first, so a reader who has it
installed sees it; everyone else gets Helvetica Neue, Arial or the system font. To use another
font, add its `@font-face` rule in `docs/_includes/head_custom.html` and put the family name
first in `font-sans`. Check the font's licence allows web use.

## Emblem and menu

- **Emblem.** The image above the site title is `seal` in `docs/_config.yml` (default `/gsa-seal.png`, a file in
  `docs/`), with its description in `seal_alt`. Leave `seal` empty for none. Its width is the `seal-size` token.
- **Light/dark button.** It sits at the top right of the menu column.
- **Menu links that leave the site.** `nav_external_links` in `_config.yml` adds them. The theme puts them after all
  the pages; `before: <menu title>` or `after: <menu title>` on an entry moves it above or below that item (done in `_includes/head_custom.html`; without
  JavaScript it just stays last).

## Home page content

The figures and cards on the home page come from `docs/_data/home.yml`. Edit that file to change
their text or links. The figures are typed by hand, so update them when the counts change.

## How it is built

```text
docs/
├── _config.yml                    theme_preset: ndstudio   (picks the preset)
├── _data/themes/*.yml             the presets (colours, fonts, sizes)
├── _data/home.yml                 home page figures and cards
├── _includes/head_custom.html     turns the preset into CSS variables (--nd-<token>) and adds the light/dark button
├── _includes/footer_custom.html   footer links
├── _includes/title.html           emblem above the site name
├── _includes/nd/cohort_report.html  body of a cohort report page (see COHORT_DATA.md)
├── _includes/nav_footer_custom.html   sidebar-bottom note on why the project is in the GSA organization
├── _includes/nd/                  stats and cards used by index.md
└── _sass/custom/                  the CSS, layered over Just the Docs; reads only var(--nd-*)
```

Rules for contributors:

- Do not write colours (`#fff`, `rgb(...)`) in `_sass/custom/`. Use `var(--nd-<token>)`. If you need
  a new value, add a token to every preset (and to `dark:` if it is a colour), and add it to the table above.
- For a rule that must differ in the dark theme, use the `@include nd-dark { ... }` mixin from
  `_sass/custom/nd/_mixins.scss`, so it follows both the button and the device setting.
- Keep text colours at WCAG AA contrast (4.5:1). `check_theme.py` tests the pairs the site uses.
- The theme is pinned in `_config.yml` (`just-the-docs@v0.12.0`). Upgrade by changing the tag and
  re-checking a few pages.

## Reference architecture pages

`docs/knowledge-graph/` is a generated snapshot of another repository's site, and its pages must not be
edited by hand. All of its pages share two files, so the preset is applied by a script instead:

```bash
python3 .github/scripts/skin_reference_architecture.py          # apply (safe to repeat)
python3 .github/scripts/skin_reference_architecture.py --check  # is it current?
```

The script appends one generated block (between `ND-SKIN BEGIN` and `ND-SKIN END`) to `assets/style.css` and
`assets/theme.js`. It maps the generator's variables onto the preset's tokens, squares the corners, flattens the
cards, and restyles the header and the light/dark button. It also makes those pages use the same saved light/dark
choice as the rest of the site, and adds a "Study guide" link back and the GSA footer note. Run it again after
changing a preset and after every refresh of the snapshot. `check_theme.py` reports if it is out of date.

| Reference architecture variable | Preset token |
|---|---|
| `--page` | `surface` |
| `--surface` | `surface-raised` |
| `--ink` | `ink` |
| `--ink-2` | `ink-soft` |
| `--muted` | `muted` |
| `--hair` | `hairline` |
| `--axis` | `axis` |
| `--accent`, `--accent-ink` | `link` |
| `--rule` | `rule` |
| `--radius` | `radius` |
| `--font`, `--mono` | `font-sans`, `font-mono` |

The graph's category colours (`--g1` to `--g8`) are left as the generator defines them: they are a validated
colour-blind-safe set, and recolouring them would hurt readability. To make the change permanent, the
generator in `Cybonto/book1` could read the same tokens; until then the script is the single place the look is applied.
