---
title: Reports
nav_order: 7
has_children: true
has_toc: false
permalink: /reports/
description: Attendance, retention and poll results for each study cohort, and how outside experts are engaging.
---

# Reports
{: .no_toc }

The reports show totals only. They never show a name, an email address, or a group small enough to point to one person.

{::nomarkdown}
{% include nd/cards.html items=site.data.home.reports %}
{:/nomarkdown}

## Across the cohorts

{::nomarkdown}
{% include nd/reports_summary.html %}
{:/nomarkdown}

Figures are added up from the cohort reports. Percentages for stayed 45 minutes or more count every attendance, once per person per session. Someone who appears in more than one report is counted in each. "Highly confident" compares poll answers from people who said they did the reading with those who said they did not.

{::nomarkdown}
{% include nd/org_chart.html %}
{:/nomarkdown}

## Expert engagement

{::nomarkdown}
{% include nd/expert_summary.html %}
{:/nomarkdown}

Counts of outside experts who replied with interest or joined the community discussion, as of {{ site.data.expert_engagement.as_of }}. The group is small, so read the counts, not only the shares.

How the numbers are produced, and the privacy rules they follow, are described in the repository file `docs/COHORT_DATA.md`.
