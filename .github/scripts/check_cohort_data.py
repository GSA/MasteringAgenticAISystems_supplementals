#!/usr/bin/env python3
"""Privacy and shape check for docs/_data/cohorts/*.json.  Run before every commit of those files.

  python3 .github/scripts/check_cohort_data.py

Fails (exit 1) if a file
  * contains an "@" anywhere (an email address),
  * has a key that is not on the allowed list (so a stray name or email column cannot slip in),
  * lists an organisation with fewer than MIN_ORG (3) people, or breaks organisations out for a cohort
    with fewer than 30 people,
  * shows a per-session poll count smaller than min_cell.
Standard library only.
"""
import json, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs" / "_data" / "cohorts"
TOP = {"id", "year", "cohort", "section", "title", "first_session", "last_session", "tiles", "sessions",
       "excluded", "retention", "tiers", "read_vs_confidence", "time_in_session", "organizations",
       "min_cell", "notes"}
TILES = {"people", "sessions", "orgs", "median_minutes", "helpful_yes", "helpful_n", "poll_response_rate"}
SESSION = {"date", "attendees", "new", "returning", "median_minutes", "stay45", "poll_people", "read_n",
           "read_yes", "conf_n", "conf_low", "conf_med", "conf_high"}
EXCLUDED = {"date", "attendees", "reason"}
ORGS = {"min_people", "listed", "other_people", "other_orgs"}
MIN_PEOPLE_FOR_ORGS = 30
MIN_ORG = 3


def main():
    problems, files = [], sorted(DATA.glob("*.json"))
    if not files:
        problems.append("no data files in docs/_data/cohorts/")
    for f in files:
        text = f.read_text(encoding="utf-8")
        name = f.name
        if "@" in text:
            problems.append(f"{name}: contains '@' (an email address?)")
        d = json.loads(text)
        if set(d) - TOP:
            problems.append(f"{name}: unexpected keys {sorted(set(d) - TOP)}")
        if set(d["tiles"]) - TILES:
            problems.append(f"{name}: unexpected tile keys {sorted(set(d['tiles']) - TILES)}")
        for s in d["sessions"]:
            if set(s) - SESSION:
                problems.append(f"{name}: unexpected session keys {sorted(set(s) - SESSION)}")
            for k in ("poll_people", "read_n", "conf_n"):
                if s.get(k) is not None and s[k] < d["min_cell"]:
                    problems.append(f"{name}: {s['date']} {k}={s[k]} is below min_cell")
        for e in d["excluded"]:
            if set(e) - EXCLUDED:
                problems.append(f"{name}: unexpected excluded keys {sorted(set(e) - EXCLUDED)}")
        o = d["organizations"]
        if o:
            if set(o) - ORGS:
                problems.append(f"{name}: unexpected organisation keys")
            if d["tiles"]["people"] < MIN_PEOPLE_FOR_ORGS:
                problems.append(f"{name}: organisations listed for a cohort of only {d['tiles']['people']} people")
            for item in o["listed"]:
                if set(item) != {"label", "people"} or item["people"] < MIN_ORG:
                    problems.append(f"{name}: organisation entry not allowed: {item}")
        print(f"  {name}: {d['tiles']['people']} people, {len(d['sessions'])} sessions, organisations {'listed' if o else 'not listed'}")
    if problems:
        print("\nCohort data check FAILED:")
        for p in problems:
            print("  -", p)
        return 1
    print("\nCohort data check passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
