---
title: Knowledge Graph
nav_order: 10.5
permalink: /architecture/
---

# Knowledge Graph
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

This is a **separate, self-contained website** with its own header and top navigation — it does not use this site's
sidebar, but it shares its look and its light/dark choice, and it makes no network calls. Once you're in it, move between sections
with its own menu, or jump straight to one:

- [Overview]({{ site.baseurl }}/knowledge-graph/index.html) — stats, layer breakdown, and full-text search over all 1,587 components
- [Layers]({{ site.baseurl }}/knowledge-graph/layers.html) — every component grouped by layer/plane
- [Variation points]({{ site.baseurl }}/knowledge-graph/variation-points.html) — the 149 components with documented alternatives
- [Profiles]({{ site.baseurl }}/knowledge-graph/profiles.html) — the 226 named architecture profiles
- [Ontology]({{ site.baseurl }}/knowledge-graph/ontology.html) — the controlled vocabulary of element kinds and relationship types
- [Sources]({{ site.baseurl }}/knowledge-graph/sources.html) — the 152 cited chapters, references, and notes

Each component also has its own page, for example `knowledge-graph/c/ToolExecutor.html`.

## Explore the graph

[Open the interactive graph explorer]({{ site.baseurl }}/knowledge-graph/explore.html){: .btn .btn-primary target="_blank" rel="noopener" }

Opens in a **new tab**, full width, with none of this site's layout — the graph needs the room, and a browser with
WebGL enabled. Hovering a node highlights its neighborhood; clicking opens a details panel. Link straight to a
component with `#<ComponentId>`, for example `…/explore.html#ToolExecutor`.

## Licensing

The graph explorer vendors two MIT-licensed libraries, `sigma.js` and `graphology`, kept offline rather than loaded
from a CDN — see `knowledge-graph/assets/vendor/NOTICE.md`. They are not covered by this repository's CC0
dedication (see [About]({{ site.baseurl }}/about/)).
