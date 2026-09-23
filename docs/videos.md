---
title: Videos
nav_order: 7
permalink: /videos/
---

# Curated videos
{: .no_toc }

Third-party YouTube resources, one file per Part, chosen to explain the same concepts from a different angle.
They are **references, not contributed material**, and the project's public-domain dedication does not cover them.

## Watch them in context

Each [Part page]({{ site.baseurl }}/curriculum/) embeds the videos for a chapter directly under that chapter's summary, in a
collapsed **Videos** section (the same way code examples are shown). In current browsers nothing loads until you open a section. Players use
YouTube's privacy-enhanced domain, and each caption gives the video's title and channel as YouTube reports them, with a link
to watch it on YouTube. 261 videos are embedded in total.

| Part | Source file | Entries | Direct links | Videos shown |
|---|---|---:|---:|---:|
| [Part 1]({{ site.baseurl }}/curriculum/part-01/) | [`Part_01_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_01_YoutubeVideos.md) | 52 | 47 | 52 |
| [Part 2]({{ site.baseurl }}/curriculum/part-02/) | [`Part_02_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_02_YoutubeVideos.md) | 25 | 20 | 25 |
| [Part 3]({{ site.baseurl }}/curriculum/part-03/) | [`Part_03_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_03_YoutubeVideos.md) | 35 | 33 | 35 |
| [Part 4]({{ site.baseurl }}/curriculum/part-04/) | [`Part_04_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md) | 39 | 32 | 39 |
| [Part 5]({{ site.baseurl }}/curriculum/part-05/) | [`Part_05_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_05_YoutubeVideos.md) | 82 | 70 | 82 |
| [Part 6]({{ site.baseurl }}/curriculum/part-06/) | [`Part_06_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_06_YoutubeVideos.md) | 21 | 20 | 21 |
| [Part 7]({{ site.baseurl }}/curriculum/part-07/) | [`Part_07_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_07_YoutubeVideos.md) | 48 | 3 | 3 |
| [Part 8]({{ site.baseurl }}/curriculum/part-08/) | [`Part_08_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md) | 23 | 2 | 2 |
| [Part 9]({{ site.baseurl }}/curriculum/part-09/) | [`Part_09_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_09_YoutubeVideos.md) | 63 | 5 | 5 |
| [Part 10]({{ site.baseurl }}/curriculum/part-10/) | [`Part_10_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md) | 67 | 2 | 2 |

"Direct links" counts unique links in the source file; entries without one are search suggestions ("Search *MIT 6.824
Spring 2020*"). Some chapters share a video, so the per-Part totals are not simply the sum of the chapters.

## The library was checked and cleaned

On 2026-09-23, every YouTube link in the source files was checked against YouTube's public oEmbed endpoint, which
reports whether a video exists and returns its real title. 93 entries were removed from `videos/`
as a result: 38 pointed at a video YouTube reports as not found, and 55 pointed at a
real video that was not the one the entry described (a title check only — not a judgment of quality). Four chapters lost
their only video and now carry a short note instead of a broken link.

What remains: 261 videos embedded on the Part pages, plus 2 valid links the uploader has
disabled from embedding, shown as plain links. The full account, including what was removed, is on the
[link check]({{ site.baseurl }}/videos/link-check/) page.

This was a one-time check and cleanup; it will go stale as videos disappear or get re-uploaded elsewhere, so re-run it
periodically rather than trusting this page indefinitely.

## Coverage

The library is uneven. Parts 1–5 carry most of the direct links; Parts 7–10 together list only 12. The project has not
otherwise verified the links, and extending and verifying the library is an open
[contribution track]({{ site.baseurl }}/contributing/).
