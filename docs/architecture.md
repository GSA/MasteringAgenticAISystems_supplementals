---
title: Reference Architecture
nav_order: 10.5
permalink: /architecture/
---

# Reference Architecture
{: .no_toc }

An interactive reference architecture for agentic AI systems, generated from a validated knowledge graph of every
architectural component described across the book's 96 chapters. Browse a searchable catalog, or explore the whole
component graph visually.

<details open markdown="block">
  <summary>On this page</summary>
  {: .text-delta }
1. TOC
{:toc}
</details>

## What it is

The reference architecture is a knowledge graph (KG) of **non-divisible architectural components** and the typed,
source-attributed relationships between them, extracted from the book's chapter text and cross-checked against
external references. It is a separate deliverable from this site's curriculum and study materials, published here as
a static website generated from that KG.

## At a glance

| Metric | Value |
|---|---:|
| Components | 1,587, of which 149 are abstract variation points |
| Relationships | 5,288, across 30 typed relationship kinds |
| Layers / planes | 13, grouped into 8 areas (Orchestration & Tools, Cognition & Memory, Knowledge & Data, Models, Infrastructure, Observability & Evaluation, Safety/Security & Governance, Experience & Human Oversight) |
| Architecture profiles | 226 named, concrete configurations |
| Cited sources | 152 (84 book chapters, 34 externally verified references, 34 reference notes) |

## Browse the site

This is a **separate, self-contained website** with its own header, top navigation, and light/dark theme toggle — it
does not use this site's sidebar or styling, and makes no network calls. Once you're in it, move between sections
with its own menu, or jump straight to one:

- [Overview]({{ site.baseurl }}/reference-architecture/index.html) — stats, layer breakdown, and full-text search over all 1,587 components
- [Layers]({{ site.baseurl }}/reference-architecture/layers.html) — every component grouped by layer/plane
- [Variation points]({{ site.baseurl }}/reference-architecture/variation-points.html) — the 149 components with documented alternatives
- [Profiles]({{ site.baseurl }}/reference-architecture/profiles.html) — the 226 named architecture profiles
- [Ontology]({{ site.baseurl }}/reference-architecture/ontology.html) — the controlled vocabulary of element kinds and relationship types
- [Sources]({{ site.baseurl }}/reference-architecture/sources.html) — the 152 cited chapters, references, and notes

Each component also has its own page, for example `reference-architecture/c/ToolExecutor.html`.

## Explore the graph

<a href="{{ site.baseurl }}/reference-architecture/explore.html" target="_blank" rel="noopener">Open the interactive graph explorer ↗</a>
{: .btn .btn-primary }

Opens in a **new tab**, full width, with none of this site's layout — the graph needs the room, and a browser with
WebGL enabled. Hovering a node highlights its neighborhood; clicking opens a details panel. Link straight to a
component with `#<ComponentId>`, for example `…/explore.html#ToolExecutor`.

## Provenance and licensing

- The pages under `reference-architecture/` are copied verbatim from the `drafts/iter3/reference_architecture/site/`
  build in `Cybonto/book1` (source commit `002b808`), not authored or hand-edited in this repository. To fix a
  problem, fix it at the source and re-copy — see that repository's `site_integration.md` for the refresh steps.
- The graph explorer vendors two MIT-licensed libraries, `sigma.js` and `graphology`, kept offline rather than loaded
  from a CDN — see `reference-architecture/assets/vendor/NOTICE.md`. They are not covered by this repository's CC0
  dedication (see [About]({{ site.baseurl }}/about/)).
