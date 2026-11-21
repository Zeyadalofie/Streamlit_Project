# Your answer - EX01 and EX02 (Role A). Write both functions in this one file.
import pandas as pd
REQUIRED_COLUMNS = ['request_id', 'opened_at', 'desk', 'region', 'request_type', 'channel', 'priority', 'attachments', 'minutes', 'staff_on_shift', 'backlog_at_open', 'wait_hours', 'status', 'satisfaction']

def load_requests(path):
    """Read, validate and sort the request file."""
    df=pd.read_csv(path ,parse_dates=["opened_at"])
    missing=[col for col in REQUIRED_COLUMNS if col not in df.columns]
    if missing:
        raise ValueError(f"missing columns:{missing}")
    df = df.sort_values("opened_at")
    df = df.reset_index(drop=True)
    return df


def filter_requests(df, regions=None, channels=None, priorities=None, date_range=None):
    """Filtered copy of a request table (see EX02)."""
    view = df.copy()
    if regions is not None:
        view = view[view["region"].isin(regions)]

    if channels is not None:
        view = view[view["channel"].isin(channels)]

    if priorities is not None:
        view = view[view["priority"].isin(priorities)]

    if date_range is not None:
        start_date, end_date = date_range

        dates = view["opened_at"].dt.date
        view = view[(dates >= start_date) &(dates <= end_date)]
    return view
