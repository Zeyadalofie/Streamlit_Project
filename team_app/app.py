# Your answer - EX09 (Team)
import streamlit as st
from dash_data import load_requests, filter_requests
from dash_metrics import compute_kpis, desk_table
from dash_charts import weekly_chart, desk_chart
from dash_findings import wait_finding, data_notes

try:                                   # optional styling; the app also runs without it (e.g. in the notebook)
    from dash_style import apply_style, sidebar_brand, evidence_pill
except ImportError:
    apply_style = sidebar_brand = lambda *args, **kwargs: None
    evidence_pill = None

st.set_page_config(page_title="Police Requests Dashboard", page_icon="🚓", layout="wide")
apply_style()

DATA_FILE = "https://raw.githubusercontent.com/hanyelshafey/ai-security-bootcamp-data/main/week_07/police_requests_team.csv"


@st.cache_data
def get_data(path):
    return load_requests(path)


df = get_data(st.session_state.get("data_path", DATA_FILE))

# ---- filter options and defaults ----
REGION_OPTIONS = sorted(df["region"].unique())
CHANNEL_OPTIONS = sorted(df["channel"].unique())
PRIORITY_OPTIONS = ["Low", "Normal", "High"]
FULL_RANGE = (df["opened_at"].min().date(), df["opened_at"].max().date())

st.session_state.setdefault("regions", list(REGION_OPTIONS))
st.session_state.setdefault("channels", list(CHANNEL_OPTIONS))
st.session_state.setdefault("priorities", list(PRIORITY_OPTIONS))
st.session_state.setdefault("dates", FULL_RANGE)


def reset_filters():
    st.session_state["regions"] = list(REGION_OPTIONS)
    st.session_state["channels"] = list(CHANNEL_OPTIONS)
    st.session_state["priorities"] = list(PRIORITY_OPTIONS)
    st.session_state["dates"] = FULL_RANGE


# ---- sidebar (widgets without defaults: values come from session_state) ----
sidebar_brand("Filters")
regions = st.sidebar.multiselect("Region", REGION_OPTIONS, key="regions")
channels = st.sidebar.multiselect("Channel", CHANNEL_OPTIONS, key="channels")
priorities = st.sidebar.multiselect("Priority", PRIORITY_OPTIONS, key="priorities")
dates = st.sidebar.date_input("Dates", key="dates", min_value=FULL_RANGE[0], max_value=FULL_RANGE[1])
st.sidebar.button("Reset filters", on_click=reset_filters, icon=":material/restart_alt:", width="stretch")

# while the user has picked only the first date, don't filter on dates yet
date_range = tuple(dates) if isinstance(dates, (tuple, list)) and len(dates) == 2 else None
view = filter_requests(df, regions, channels, priorities, date_range)

# ---- page ----
notes = data_notes(view, df)

st.title("Police Requests Dashboard")
st.caption(notes[0])

if view.empty:
    st.warning("No requests match the filters.")

kpis = compute_kpis(view)
for col, (label, value) in zip(st.columns(5), kpis.items()):
    col.metric(label, value)

st.write("")   # breathing room between the indicators and the tabs
overview, desks, findings, data = st.tabs(["Overview", "Desks", "Findings", "Data"])

with overview:
    fig = weekly_chart(view)
    if fig is None:
        st.info("No complete week in this selection.")
    else:
        with st.container(border=True):
            st.pyplot(fig, width=760)

with desks:
    st.dataframe(desk_table(view), column_config={     # display formats only: the data are unchanged
        "requests": st.column_config.NumberColumn("Requests", format="%d"),
        "open_rate": st.column_config.NumberColumn("Open rate", format="percent"),
        "median_minutes": st.column_config.NumberColumn("Median minutes", format="%.1f"),
        "median_wait": st.column_config.NumberColumn("Median wait (h)", format="%.1f"),
        "survey_rate": st.column_config.NumberColumn("Survey answered", format="percent"),
    }, width="stretch")
    fig = desk_chart(view)
    if fig is not None:
        with st.container(border=True):
            st.pyplot(fig, width=760)

with findings:
    finding = wait_finding(view)
    if finding is None:
        st.info("Not enough data for a finding.")
    else:
        low, high = finding["ci"]
        with st.container(border=True):
            if evidence_pill is not None:
                st.html(f"<div style='margin-bottom:-6px'>Evidence {evidence_pill(finding['evidence'])}</div>")
            st.markdown(finding["headline"])
            st.caption(f"Evidence: {finding['evidence']} | 95% interval for the difference: "
                       f"{low:.2f} to {high:.2f} h | smaller desk n = {finding['n_min']}")

with data:
    st.dataframe(view, height=420)
    st.download_button("Download filtered CSV", view.to_csv(index=False),
                       file_name="filtered_requests.csv", mime="text/csv", icon=":material/download:")

with st.expander("Method and data notes"):
    for note in notes[1:]:
        st.markdown(note)
