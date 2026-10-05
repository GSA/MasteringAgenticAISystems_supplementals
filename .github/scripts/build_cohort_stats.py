#!/usr/bin/env python3
"""Builds the aggregate numbers behind the cohort report pages.

  python3 .github/scripts/build_cohort_stats.py --raw "/path/to/CPE/rawdata"
  python3 .github/scripts/build_cohort_stats.py --raw ... --dry-run     (print a summary, write nothing)

Reads, for each cohort listed in COHORTS, two Zoom exports that were combined by hand:
  <raw>/<folder>/participants_combined.csv   Meeting Date, Name, Email, Total duration (minutes), Guest
  <raw>/<folder>/poll_combined.csv           Meeting Date, Poll Name, Question, User Name, Email, ..., Answer
and writes docs/_data/cohorts/<slug>.json.

THE JSON HOLDS AGGREGATES ONLY. No name or email address is ever written. Organisations (email
domains) appear only in a cohort with at least MIN_PEOPLE_FOR_ORGS people, and only when at least
MIN_CELL people share a domain. Poll results for a session are dropped when fewer than MIN_CELL
people answered. .github/scripts/check_cohort_data.py re-checks the output before you commit it.

Rules for what counts (so every cohort is treated the same):
  * A person is an email address, lower-cased. Rows without an email are not counted as people.
  * A person's time in a session is the sum of their rows for that date, capped at 90 minutes
    (the same person can appear twice, for example on two devices).
  * A date is a session that was held if the median time in session is at least 10 minutes.
    Other dates (a cancelled or failed session) are listed under "excluded" and left out.
  * "New" means the first held session a person attended; "returning" means any later one.
  * The reading question was worded two ways; both are treated as one question.

Standard library only.
"""
import argparse, csv, json, statistics, sys
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUT_DIR = ROOT / "docs" / "_data" / "cohorts"

COHORTS = [
    {"slug": "2026-cohort-1-section-2", "year": 2026, "cohort": 1, "section": 2,
     "folder": "Cohort 1 Section 2 Zoom Reports"},
    {"slug": "2026-cohort-2-section-1", "year": 2026, "cohort": 2, "section": 1,
     "folder": "Cohort 2 Section 1 Zoom Reports"},
]

MIN_CELL = 5                # smallest group shown for an organisation or a per-session poll result
MIN_PEOPLE_FOR_ORGS = 30    # below this, organisations are not broken out at all
MIN_PAIRS = 10              # smallest group for "did the reading" versus confidence
HELD_MEDIAN_MINUTES = 10
CAP_MINUTES = 90
STAY_MINUTES = 45
HIST_BINS = [(0, 10, "under 10"), (10, 30, "10 to 29"), (30, 45, "30 to 44"),
             (45, 60, "45 to 59"), (60, 75, "60 to 74"), (75, 999, "75 or more")]
TIERS = [(0, 0.25, "25% or fewer"), (0.25, 0.5, "26 to 50%"), (0.5, 0.75, "51 to 75%"), (0.75, 1.01, "more than 75%")]
# Plain names for the few domains that can be listed (domain -> name). Others show the domain.
DOMAIN_NAMES = {"usda.gov": "USDA", "va.gov": "Veterans Affairs", "gsa.gov": "GSA", "irs.gov": "IRS",
                "tsa.dhs.gov": "TSA", "epa.gov": "EPA", "noaa.gov": "NOAA", "sec.gov": "SEC",
                "fda.hhs.gov": "FDA", "cms.hhs.gov": "CMS", "uspto.gov": "USPTO", "faa.gov": "FAA",
                "dot.gov": "DOT", "gao.gov": "GAO", "nih.gov": "NIH", "bls.gov": "BLS",
                "hq.doe.gov": "DOE"}


def read_rows(path):
    with open(path, encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def parse_date(s):
    return datetime.strptime(s.strip(), "%m/%d/%Y").date()


def pct(a, b):
    return round(a / b, 4) if b else None


def build(cohort, raw):
    folder = Path(raw) / cohort["folder"]
    prows = read_rows(folder / "participants_combined.csv")
    qrows = read_rows(folder / "poll_combined.csv")

    # --- people and time in session -------------------------------------------------------
    minutes = defaultdict(int)              # (date, email) -> minutes
    no_email = 0
    for r in prows:
        em = (r.get("Email") or "").strip().lower()
        if not em:
            no_email += 1
            continue
        minutes[(parse_date(r["Meeting Date"]), em)] += int(float(r["Total duration (minutes)"] or 0))
    per_date = defaultdict(dict)
    for (d, em), m in minutes.items():
        per_date[d][em] = min(m, CAP_MINUTES)

    held, excluded = [], []
    for d in sorted(per_date):
        med = statistics.median(per_date[d].values())
        if med >= HELD_MEDIAN_MINUTES:
            held.append(d)
        else:
            excluded.append({"date": d.isoformat(), "attendees": len(per_date[d]),
                             "reason": "Median time in the meeting was %d minutes, so it was not a held session." % med})
    first_seen = {}
    for d in held:
        for em in per_date[d]:
            first_seen.setdefault(em, d)
    people = set(first_seen)

    # --- polls ------------------------------------------------------------------------------
    polls = defaultdict(lambda: defaultdict(list))      # date -> kind -> [(email, answer)]
    for r in qrows:
        d = parse_date(r["Meeting Date"])
        if d not in held:
            continue
        name = r["Poll Name"].lower()
        kind = "conf" if "confidence" in name else "read" if "assignment" in name else "help" if "helpful" in name else None
        if kind:
            polls[d][kind].append(((r.get("Email Address") or "").strip().lower(), (r["Answer"] or "").strip()))

    sessions = []
    base = set(per_date[held[0]]) if held else set()
    retention = []
    for i, d in enumerate(held):
        att = per_date[d]
        new = sum(1 for em in att if first_seen[em] == d)
        row = {"date": d.isoformat(), "attendees": len(att), "new": new, "returning": len(att) - new,
               "median_minutes": statistics.median(att.values()),
               "stay45": pct(sum(1 for m in att.values() if m >= STAY_MINUTES), len(att))}
        p = polls.get(d, {})
        responders = {em for kind in p.values() for em, _ in kind}
        row["poll_people"] = len(responders) if len(responders) >= MIN_CELL else None
        rd = p.get("read", [])
        row["read_n"] = len(rd) if len(rd) >= MIN_CELL else None
        row["read_yes"] = sum(1 for _, a in rd if a == "Yes") if len(rd) >= MIN_CELL else None
        cf = p.get("conf", [])
        c = Counter(a for _, a in cf)
        ok = len(cf) >= MIN_CELL
        row["conf_n"] = len(cf) if ok else None
        for key, label in (("conf_low", "Low"), ("conf_med", "Medium"), ("conf_high", "High")):
            row[key] = c.get(label, 0) if ok else None
        sessions.append(row)
        if i > 0 and base:
            retention.append({"date": d.isoformat(), "share": pct(len(base & set(att)), len(base))})

    # --- engagement tiers ------------------------------------------------------------------
    counts = Counter(em for d in held for em in per_date[d])
    tier_counts = []
    for lo, hi, label in TIERS:
        n = sum(1 for em in people if lo < counts[em] / len(held) <= hi)
        tier_counts.append({"label": label, "people": n})

    # --- reading versus confidence (same person, same day) --------------------------------
    groups = {"read": Counter(), "noread": Counter()}
    for d in held:
        rd = {em: a for em, a in polls.get(d, {}).get("read", [])}
        for em, a in polls.get(d, {}).get("conf", []):
            if em in rd:
                groups["read" if rd[em] == "Yes" else "noread"][a] += 1
    readconf = {}
    for k, c in groups.items():
        n = sum(c.values())
        readconf[k] = ({"n": n, "low": c.get("Low", 0), "med": c.get("Medium", 0), "high": c.get("High", 0)}
                       if n >= MIN_PAIRS else None)

    # --- time in session ---------------------------------------------------------------------
    allm = [m for d in held for m in per_date[d].values()]
    hist = [{"label": label, "count": sum(1 for m in allm if lo <= m < hi)} for lo, hi, label in HIST_BINS]

    # --- organisations -----------------------------------------------------------------------
    orgs = None
    domain_of = {em: em.split("@")[-1] for em in people}
    n_orgs = len(set(domain_of.values()))
    if len(people) >= MIN_PEOPLE_FOR_ORGS:
        by = Counter(domain_of.values())
        listed = [{"label": DOMAIN_NAMES.get(dom, dom), "people": n} for dom, n in by.most_common() if n >= MIN_CELL]
        rest = [(dom, n) for dom, n in by.items() if n < MIN_CELL]
        orgs = {"min_people": MIN_CELL, "listed": listed,
                "other_people": sum(n for _, n in rest), "other_orgs": len(rest)}

    # --- headline numbers --------------------------------------------------------------------
    help_all = [a for d in held for _, a in polls.get(d, {}).get("help", [])]
    rates = []
    for d in held:
        resp = {em for kind in polls.get(d, {}).values() for em, _ in kind}
        if resp:
            rates.append(len(resp) / len(per_date[d]))
    tiles = {"people": len(people), "sessions": len(held), "orgs": n_orgs,
             "median_minutes": statistics.median(allm) if allm else None,
             "helpful_yes": sum(1 for a in help_all if a == "Yes"), "helpful_n": len(help_all),
             "poll_response_rate": round(statistics.mean(rates), 4) if rates else None}

    notes = []
    if excluded:
        notes.append("%d meeting date(s) were left out because they were not held sessions." % len(excluded))
    if no_email:
        notes.append("%d attendance rows had no email address, so they are not counted as people." % no_email)
    stray = sorted({parse_date(r["Meeting Date"]).isoformat() for r in qrows if parse_date(r["Meeting Date"]) not in held})
    if stray:
        notes.append("Poll answers dated %s were left out because there is no attendance record for a held session that day." % ", ".join(stray))
    missing = [s["date"] for s in sessions if s["conf_n"] is None]
    if missing:
        notes.append("No usable poll results for: %s (no poll file, or fewer than %d answers)." % (", ".join(missing), MIN_CELL))
    notes.append("Poll results come from people who were in the meeting when a poll opened and chose to answer.")
    notes.append("Percentages in a small cohort move a lot with one person; read the counts.")

    return {
        "id": cohort["slug"], "year": cohort["year"], "cohort": cohort["cohort"], "section": cohort["section"],
        "title": "%d Cohort %d Section %d" % (cohort["year"], cohort["cohort"], cohort["section"]),
        "first_session": held[0].isoformat() if held else None, "last_session": held[-1].isoformat() if held else None,
        "tiles": tiles, "sessions": sessions, "excluded": excluded,
        "retention": {"opening_group": len(base), "points": retention},
        "tiers": {"sessions_held": len(held), "people": len(people), "bins": tier_counts},
        "read_vs_confidence": readconf, "time_in_session": hist, "organizations": orgs,
        "min_cell": MIN_CELL, "notes": notes,
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", required=True, help="folder that holds the cohort report folders")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in COHORTS:
        data = build(c, a.raw)
        t = data["tiles"]
        print("%s: %d people, %d sessions held, %d excluded, %d organisations, orgs listed: %s" % (
            data["title"], t["people"], t["sessions"], len(data["excluded"]), t["orgs"],
            "yes" if data["organizations"] else "no"))
        if not a.dry_run:
            (OUT_DIR / (c["slug"] + ".json")).write_text(
                json.dumps(data, indent=1, sort_keys=True, ensure_ascii=True) + "\n", encoding="utf-8")
    if not a.dry_run:
        print("Wrote %s. Now run: python3 .github/scripts/check_cohort_data.py" % OUT_DIR.relative_to(ROOT))
    return 0


if __name__ == "__main__":
    sys.exit(main())
