# Your answer - EX03 and EX04 (Role B). Write both functions in this one file.
import pandas as pd

def compute_kpis(df):
    """Five KPI display strings (see EX03)."""
    if df.empty:
        return {
            "Requests": "0",
            "Median minutes": "—",
            "Open share": "—",
            "Median wait": "—",
            "Survey answered": "—",
        }

    n = len(df)
    median_minutes = df["minutes"].median()
    open_share = df["status"].eq("open").mean()
    median_wait = df["wait_hours"].median()
    answered_share = df["satisfaction"].notna().mean()

    return {
        "Requests": f"{n:,}",
        "Median minutes": f"{median_minutes:.1f} min",
        "Open share": f"{open_share:.1%}",
        "Median wait": f"{median_wait:.1f} h",
        "Survey answered": f"{answered_share:.0%}",
    }

def desk_table(df):
    """One row per desk (see EX04)."""
    COLUMNS = ["requests", "open_rate", "median_minutes", "median_wait", "survey_rate"]
    if df.empty:
      return pd.DataFrame(columns=COLUMNS, index=pd.Index([], name="desk"))

    table = df.groupby('desk').agg(
        requests=("request_id", "size"),
        open_rate=("status", lambda s: s.eq("open").mean()),
        median_minutes=("minutes", "median"),
        median_wait=("wait_hours", "median"),
        survey_rate=("satisfaction", lambda s: s.notna().mean()),
    )

    table = (table.reset_index().sort_values(["requests", "desk"], ascending=[False, True]).set_index("desk"))
    return table[COLUMNS]
