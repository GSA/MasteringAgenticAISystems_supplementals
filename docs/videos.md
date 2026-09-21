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

Each [Part page]({% link curriculum/index.md %}) embeds the videos for a chapter directly under that chapter's summary, in a
collapsed **Videos** section (the same way code examples are shown). Nothing loads until you open a section. Players use
YouTube's privacy-enhanced domain, and each caption gives the video's title and channel as YouTube reports them, with a link
to watch it on YouTube. 261 videos are embedded in total.

| Part | Source file | Entries | Direct links | Videos shown |
|---|---|---:|---:|---:|
| [Part 1]({% link curriculum/part-01.md %}) | [`Part_01_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_01_YoutubeVideos.md) | 59 | 54 | 52 |
| [Part 2]({% link curriculum/part-02.md %}) | [`Part_02_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_02_YoutubeVideos.md) | 32 | 23 | 25 |
| [Part 3]({% link curriculum/part-03.md %}) | [`Part_03_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_03_YoutubeVideos.md) | 80 | 75 | 35 |
| [Part 4]({% link curriculum/part-04.md %}) | [`Part_04_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_04_YoutubeVideos.md) | 54 | 45 | 39 |
| [Part 5]({% link curriculum/part-05.md %}) | [`Part_05_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_05_YoutubeVideos.md) | 98 | 86 | 82 |
| [Part 6]({% link curriculum/part-06.md %}) | [`Part_06_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_06_YoutubeVideos.md) | 22 | 20 | 21 |
| [Part 7]({% link curriculum/part-07.md %}) | [`Part_07_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_07_YoutubeVideos.md) | 48 | 3 | 3 |
| [Part 8]({% link curriculum/part-08.md %}) | [`Part_08_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_08_YoutubeVideos.md) | 23 | 2 | 2 |
| [Part 9]({% link curriculum/part-09.md %}) | [`Part_09_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_09_YoutubeVideos.md) | 64 | 6 | 5 |
| [Part 10]({% link curriculum/part-10.md %}) | [`Part_10_YoutubeVideos.md`]({{ site.repo_blob }}/videos/Part_10_YoutubeVideos.md) | 68 | 3 | 2 |

"Direct links" counts unique links in the source file; entries without one are search suggestions ("Search *MIT 6.824
Spring 2020*"). Some chapters share a video, so the per-Part totals are not simply the sum of the chapters.

## Not every link is embedded

On 2026-09-21, every YouTube link in the source files was checked against YouTube's public oEmbed endpoint. Of
267 unique YouTube links, 229 are available, 36 were not found (removed, private, or a wrong ID), and 2
cannot be embedded because the uploader disabled it. Among the available ones, some are a **different video from the one the
entry describes**; those are left out too. Embedded videos are the ones whose YouTube title matches the entry's, plus a
few reviewed by hand. Everything left out, with the reason, is on the [link check]({% link video-link-check.md %}) page.

This checks that a link resolves and that its title looks right, not that the video is a good explanation of the chapter, and
it will go stale as videos come and go.

## Coverage

The library is uneven. Parts 1–5 carry most of the direct links; Parts 7–10 together list only 14. The project has not
otherwise verified the links, and extending and verifying the library is an open
[contribution track]({% link contributing.md %}).
