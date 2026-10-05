# Styling the site

The site's look comes from one small file of values, called a **preset**. You can change
colours, fonts, corner shape and heading sizes without writing any CSS.

The default preset, `ndstudio`, is a light, flat, high-contrast style inspired by
[ndstudio.gov](https://ndstudio.gov/). It borrows the design language only (colour, spacing,
type scale). It uses no logos, images or font files from that site.

## Re-skin in five minutes

1. Copy `docs/_data/themes/ndstudio.yml` to `docs/_data/themes/<yourname>.yml`.
2. Edit the values under `tokens:`. Colours must be six-digit hex such as `"#1a1a1a"`.
3. In `docs/_config.yml`, set `theme_preset: <yourname>`.
4. Run `python3 .github/scripts/check_theme.py`. It fails if any text colour is hard to read.
5. Commit. GitHub Pages rebuilds the site.

Two presets ship with the repository: `ndstudio` (the default) and `classic` (an approximation
of the stock Just the Docs look). Set `theme_preset: classic` to go back.

## What each token controls

| Token | Controls |
|---|---|
| `surface` | Page and sidebar background |
| `surface-raised` | Cards, code blocks, search box, table row hover |
| `ink` | Headings and main text |
| `ink-soft` | Navigation links, secondary text |
| `muted` | Captions, table headers, breadcrumbs |
| `hairline` | Thin dividers |
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

## Fonts

The ndstudio.gov site uses PP Neue Montreal, which is a licensed commercial font. This
repository does not include it. The `ndstudio` preset lists it first, so a reader who has it
installed sees it; everyone else gets Helvetica Neue, Arial or the system font. To use another
font, add its `@font-face` rule in `docs/_includes/head_custom.html` and put the family name
first in `font-sans`. Check the font's licence allows web use.

## Home page content

The figures and cards on the home page come from `docs/_data/home.yml`. Edit that file to change
their text or links. The figures are typed by hand, so update them when the counts change.

## How it is built

```text
docs/
├── _config.yml                    theme_preset: ndstudio   (picks the preset)
├── _data/themes/*.yml             the presets (colours, fonts, sizes)
├── _data/home.yml                 home page figures and cards
├── _includes/head_custom.html     turns the preset into CSS variables, named --nd-<token>
├── _includes/footer_custom.html   footer links
├── _includes/nd/                  stats and cards used by index.md
└── _sass/custom/                  the CSS, layered over Just the Docs; reads only var(--nd-*)
```

Rules for contributors:

- Do not write colours (`#fff`, `rgb(...)`) in `_sass/custom/`. Use `var(--nd-<token>)`. If you need
  a new value, add a token to every preset, and add it to the table above.
- Keep text colours at WCAG AA contrast (4.5:1). `check_theme.py` tests the pairs the site uses.
- The theme is pinned in `_config.yml` (`just-the-docs@v0.12.0`). Upgrade by changing the tag and
  re-checking a few pages.

## Reference architecture pages

`docs/reference-architecture/` is a generated snapshot with its own stylesheet (`assets/style.css`) and
is **not yet covered by these presets**. Do not edit it here: fix the generator in the `Cybonto/book1`
repository and re-copy (see `docs/README.md`, "Reference Architecture").

To bring it into line, have the generator emit its `:root` variables from the same preset. The mapping:

| Reference architecture variable | Preset token |
|---|---|
| `--page` | `surface` |
| `--surface` | `surface-raised` |
| `--ink` | `ink` |
| `--ink-2` | `ink-soft` |
| `--muted` | `muted` |
| `--hair` | `hairline` |
| `--accent`, `--accent-ink` | `link`, `link-hover` |
| `--radius` | `radius` |
| `--font`, `--mono` | `font-sans`, `font-mono` |

Leave the graph's category colours (`--g1` to `--g8`) as they are: they are a validated colour-blind-safe
set, and recolouring them would hurt readability. Its dark theme and toggle can stay until a dark preset
exists.
