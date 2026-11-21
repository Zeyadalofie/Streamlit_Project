# AI extension page: wait-time predictor with an honest scorecard (not part of EX01-EX10)
import streamlit as st
from dash_data import load_requests
from dash_ai import (train_models, predict_wait, importance_chart, unusual_requests,
                    learning_curves, learning_animation, CHOSEN_TREES)

st.set_page_config(layout="wide")


DATA_FILE = "https://raw.githubusercontent.com/hanyelshafey/ai-security-bootcamp-data/main/week_07/police_requests_team.csv"


@st.cache_data
def get_data(path):
    return load_requests(path)


@st.cache_resource
def get_models(path):
    return train_models(get_data(path))


@st.cache_resource
def get_curves(path):
    return learning_curves(get_data(path))


path = st.session_state.get("data_path", DATA_FILE)
df = get_data(path)
with st.spinner("Training the models (only on the first visit)..."):
    bundle = get_models(path)
    curves = get_curves(path)

st.title("AI Wait Predictor")
st.caption("Gradient boosting trained on the whole quarter (not on the dashboard filters). "
           "Synthetic training data: not for operational decisions.")

if bundle is None:
    st.warning("Not enough requests with a recorded wait to train a reliable model.")
    st.stop()

# ---- 0. animation: watch the model learn ----
st.subheader("Watch the model learn")
try:
    animation = learning_animation(curves)
except ModuleNotFoundError:
    animation = None
if animation is None:
    st.warning("The animation needs Plotly. In a terminal run: python -m pip install plotly, then restart the app.")
else:
    st.plotly_chart(animation, width="stretch")
best = curves["best_tree"]
st.caption(f"Press Play. Gradient boosting adds one small tree at a time, each correcting the errors left by the "
           f"previous ones. Left: the average prediction at each backlog level, with its 10%-90% range. "
           f"Right: grey = error on the {bundle['n_train']} training requests, red = error on "
           f"{bundle['n_test']} requests the model never saw. The red line is lowest near tree {best}, then "
           f"creeps up while the grey line keeps falling: that gap is overfitting. "
           f"We use {CHOSEN_TREES} trees with shallow depth to stay near the best point.")

# ---- 1. scorecard: is the model worth using? ----
st.subheader("1. Is the model better than a simple guess?")
c1, c2, c3, c4 = st.columns(4)
c1.metric("Model error (MAE)", f"{bundle['mae_model']:.2f} h")
c2.metric("Simple guess error", f"{bundle['mae_baseline']:.2f} h")
c3.metric("Improvement", f"{bundle['improvement']:.0%}")
c4.metric("80% range hit rate", f"{bundle['coverage']:.0%}")
st.caption(f"Measured on {bundle['n_test']} requests the model never saw (trained on {bundle['n_train']}). "
           "Simple guess = the median wait for every request. "
           "The 80% range should contain about 80% of real waits; below 80% means the range is a bit too narrow.")

# ---- 2. what-if predictor ----
st.subheader("2. Predict the wait for a request")
cats = bundle["categories"]
b_lo, b_hi = bundle["ranges"]["backlog_at_open"]
s_lo, s_hi = bundle["ranges"]["staff_on_shift"]

left, right = st.columns([1, 1])
with left:
    # number boxes, not sliders: Streamlit sliders run backwards when the browser language is Arabic
    backlog = st.number_input("Backlog at open (requests waiting)", min_value=b_lo, max_value=b_hi,
                              value=int(df["backlog_at_open"].median()), step=1)
    staff = st.number_input("Staff on shift", min_value=s_lo, max_value=s_hi,
                            value=int(df["staff_on_shift"].median()), step=1)
    priority = st.selectbox("Priority", cats["priority"], index=cats["priority"].index("Normal")
                            if "Normal" in cats["priority"] else 0)
    request_type = st.selectbox("Request type", cats["request_type"])
    desk = st.selectbox("Desk", cats["desk"])

low, mid, high = predict_wait(bundle, {"backlog_at_open": backlog, "staff_on_shift": staff, "priority": priority,
                                       "request_type": request_type, "desk": desk})
with right:
    st.metric("Predicted wait", f"{mid:.1f} h")
    st.markdown(f"Likely between **{low:.1f} h** and **{high:.1f} h** (80% range).")
    st.info("This describes patterns in past requests, not cause and effect. "
            "It does not say what would happen if staffing were changed.")

# ---- 3. explanation ----
st.subheader("3. What drives the prediction?")
st.pyplot(importance_chart(bundle), width=800)
st.caption("Permutation importance: how much the error grows when one input is shuffled at random. "
           "A bar near 0 means the model barely uses that input.")

# ---- 4. unusual waits ----
st.subheader("4. Requests that waited much longer than expected")
st.dataframe(unusual_requests(bundle, df), hide_index=True)
st.caption("Expected wait comes from a model that never saw that request (5-fold cross-validation). "
           "A long wait during a busy shift is normal; these waited long even for their conditions, "
           "so they are worth a review.")

with st.expander("Method and limitations"):
    st.markdown("- Inputs: backlog at open, staff on shift, priority, request type, desk. "
                "Region is left out because each desk belongs to one region.")
    st.markdown("- Three models with quantile loss give the 10%, 50% and 90% predictions; "
                "shallow trees (depth 3) because 600 requests are too few for a complex model.")
    st.markdown("- Correlation is not causation: shifts with more staff may also differ in other ways.")
    st.markdown("- One quarter of synthetic data; a real deployment would need retraining and monitoring.")
