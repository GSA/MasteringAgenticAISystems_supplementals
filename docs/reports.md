# Cohort reports

The pages under **Reports** show attendance and poll results for each study cohort. This file explains where the numbers come from, the privacy rules they follow, and how to add a cohort. It is excluded from the built site.

## What is published

For each cohort and section, one page (for example `reports/2026-cohort-1-section-2.md`) with these charts: headline numbers; attendance per session; whether the opening group kept coming; how many sessions people attended; whether people did the reading; confidence by session; confidence with and without the reading; time in session; and where people work (large cohorts only). Every chart has a table view.

## Where the numbers come from

Two Zoom exports per cohort, combined by hand into `participants_combined.csv` and `poll_combined.csv` in the cohort's report folder. These files hold names and email addresses. **They never go in this repository.**

A script turns them into totals:

```bash
python3 .github/scripts/build_cohort_stats.py --raw "/path/to/the/folder/that/holds/the/cohort/folders"
python3 .github/scripts/check_cohort_data.py
```

The first writes `docs/_data/cohorts/<id>.json`. The second checks it. Commit only the JSON files, never the CSVs.

## Privacy rules (enforced by the scripts)

- The JSON holds totals only: no name, no email address, no per-person row.
- Organizations (email domains) are listed only for a cohort of 30 or more people, and only when 5 or more people share one. Everyone else is shown as a single "everyone else" figure.
- A poll result for one session is dropped when fewer than 5 people answered.
- "Did the reading" against confidence is shown only when each group has at least 10 answers.
- `check_cohort_data.py` fails if the JSON contains an `@`, a key that is not on its list, a listed organization with fewer than 5 people, or organizations for a small cohort.

## Counting rules

- A person is an email address, lower-cased. Rows without an email are not counted as people.
- A person's time in a session is the sum of their rows for that date, capped at 90 minutes.
- A date is a held session if the median time in the meeting is at least 10 minutes. Other dates are listed on the page as left out.
- "First session" means the first held session a person attended.
- The two wordings of the reading question are treated as one.
- Poll answers on a date with no held session are left out, and the page says so.

## Add a cohort

1. Put its two combined CSVs in a report folder next to the others.
2. Add an entry to `COHORTS` in `.github/scripts/build_cohort_stats.py` (id, year, cohort, section, folder).
3. Run the two commands above.
4. Copy a page in `docs/reports/`, change its title, `permalink`, `nav_order` and `cohort:` (the id).
5. Add a card for it in `docs/_data/home.yml`, under `reports:`.

## Look and feel

Charts are drawn by `docs/assets/js/cohort-charts.js` from the data embedded in the page, with no libraries and no network requests. Colors come from the `viz-*` tokens in the theme presets (see `STYLE.md`), so they follow the light and dark themes.
