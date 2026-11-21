"""Shared tools from Day 4 for every role (provided, do not edit)."""
import numpy as np
import pandas as pd

HIGHLIGHT, GREY, PURPLE = "#C0392B", "#B0B0B0", "#4F29B7"

def week_table(df):
    """Requests per Monday-Sunday week (index = Sunday) and whether the week is complete."""
    weekly = df.set_index("opened_at").resample("W-SUN").size().rename("requests").to_frame()
    first_day, last_day = df["opened_at"].min().normalize(), df["opened_at"].max().normalize()
    weekly["complete"] = (weekly.index - pd.Timedelta(days=6) >= first_day) & (weekly.index <= last_day)
    return weekly

def diff_ci(a, b, stat=np.median, n_boot=2000, level=0.95, seed=0):
    """Bootstrap interval for stat(a) - stat(b). Missing values are dropped."""
    a = np.asarray(a, dtype=float); a = a[~np.isnan(a)]
    b = np.asarray(b, dtype=float); b = b[~np.isnan(b)]
    rng = np.random.default_rng(seed)
    ia = rng.integers(0, len(a), size=(n_boot, len(a)))
    ib = rng.integers(0, len(b), size=(n_boot, len(b)))
    diffs = stat(a[ia], axis=1) - stat(b[ib], axis=1)
    low, high = np.quantile(diffs, [(1 - level) / 2, (1 + level) / 2])
    return float(low), float(high)

def evidence_label(ci, n_min):
    """weak: interval includes 0; moderate: excludes 0 but a group < 30; strong: excludes 0, both >= 30."""
    if ci[0] <= 0 <= ci[1]:
        return "weak"
    return "strong" if n_min >= 30 else "moderate"
