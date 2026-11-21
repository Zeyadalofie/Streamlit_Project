# Your answer - EX07 and EX08 (Role D). Write both functions in this one file.
from dash_tools import diff_ci, evidence_label
import pandas as pd

def wait_finding(df, n_boot=1000, seed=0):
    """Slowest versus fastest desk by median wait, with an evidence label, or None (see EX07)."""
    counts = df["desk"].value_counts()
    big_desks = counts[counts >= 10].index          # desks with at least 10 requests
    if len(big_desks) < 2:
        return None

    sub = df[df["desk"].isin(big_desks)]
    medians = sub.groupby("desk")["wait_hours"].median()
    slow, fast = medians.idxmax(), medians.idxmin()

    slow_waits = sub.loc[sub["desk"] == slow, "wait_hours"]
    fast_waits = sub.loc[sub["desk"] == fast, "wait_hours"]

    ci = diff_ci(slow_waits, fast_waits, n_boot=n_boot, seed=seed)
    n_min = int(min(counts[slow], counts[fast]))

    headline = (f"{slow} had the longest median wait ({medians[slow]:.1f} h) "
                f"and {fast} the shortest ({medians[fast]:.1f} h).")
    return {"headline": headline, "evidence": evidence_label(ci, n_min), "ci": ci, "n_min": n_min}


def data_notes(view, df):
    """Four notes for the reader (EX08, fixed)."""
    start, end = df["opened_at"].min(), df["opened_at"].max()

    if view.empty:
        survey_note = "No requests shown, so no survey rate."
    else:
        rate = view["satisfaction"].notna().mean()
        survey_note = f"Satisfaction answered for {rate:.0%} of shown requests"

    return [
        f"Showing {len(view):,} of {len(df):,} requests",
        f"Data from {start:%d %b %Y} to {end:%d %b %Y} (whole file)",
        survey_note,
        "Synthetic training data: not for operational decisions.",
    ]
