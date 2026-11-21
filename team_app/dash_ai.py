"""AI extension: predict wait_hours with an 80% range, explain the drivers, flag unusual waits.

Gradient boosting with quantile loss (10%, 50%, 90%). Every number shown on the AI page is
measured on requests the model did not see: a 25% test split for the scorecard and 5-fold
out-of-fold predictions for the unusual-wait table.
"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold, train_test_split

from dash_tools import HIGHLIGHT, GREY, PURPLE

FEATURES = ["backlog_at_open", "staff_on_shift", "priority", "request_type", "desk"]
CATEGORICAL = ["priority", "request_type", "desk"]
QUANTILES = (0.1, 0.5, 0.9)
MIN_ROWS = 100           
FEATURE_NAMES = {
    "backlog_at_open": "Backlog at open",
    "staff_on_shift": "Staff on shift",
    "priority": "Priority",
    "request_type": "Request type",
    "desk": "Desk",
}


def _model(quantile, seed):
    # shallow trees and a small learning rate: 600 rows are too few for a deep model
    return HistGradientBoostingRegressor(loss="quantile", quantile=quantile, categorical_features="from_dtype",
                                         max_depth=3, learning_rate=0.05, max_iter=300, random_state=seed)


def prepare_features(df, categories):
    """Model input table with fixed category lists, so training and prediction use the same codes."""
    X = df[FEATURES].copy()
    for col in CATEGORICAL:
        X[col] = pd.Categorical(X[col], categories=categories[col])
    return X


def _fit_all(X, y, seed):
    return {q: _model(q, seed).fit(X, y) for q in QUANTILES}


def train_models(df, seed=0):
    """Train, score and explain the model. Returns a dict, or None when there are too few rows."""
    data = df.dropna(subset=FEATURES + ["wait_hours"])
    if len(data) < MIN_ROWS:
        return None

    categories = {col: sorted(data[col].unique()) for col in CATEGORICAL}
    X, y = prepare_features(data, categories), data["wait_hours"]

    # 1. honest scorecard on a held-out 25%
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)
    test_models = _fit_all(X_tr, y_tr, seed)
    pred = {q: m.predict(X_te) for q, m in test_models.items()}
    mae_model = mean_absolute_error(y_te, pred[0.5])
    mae_baseline = mean_absolute_error(y_te, np.full(len(y_te), y_tr.median()))
    coverage = float(((y_te >= pred[0.1]) & (y_te <= pred[0.9])).mean())

    # 2. which inputs matter: shuffle one column at a time on the test set
    imp = permutation_importance(test_models[0.5], X_te, y_te, n_repeats=10, random_state=seed,
                                 scoring="neg_mean_absolute_error")
    importance = pd.Series(imp.importances_mean, index=FEATURES).sort_values(ascending=False)

    # 3. out-of-fold predictions: each request is predicted by a model that never saw it
    oof = pd.Series(np.nan, index=data.index)
    for tr, te in KFold(n_splits=5, shuffle=True, random_state=seed).split(X):
        m = _model(0.5, seed).fit(X.iloc[tr], y.iloc[tr])
        oof.iloc[te] = m.predict(X.iloc[te])

    # 4. final models on all rows, used for the what-if predictor
    return {
        "models": _fit_all(X, y, seed),
        "categories": categories,
        "mae_model": float(mae_model),
        "mae_baseline": float(mae_baseline),
        "improvement": float(1 - mae_model / mae_baseline),
        "coverage": coverage,
        "importance": importance,
        "oof_pred": oof,
        "n_train": len(X_tr),
        "n_test": len(X_te),
        "ranges": {col: (int(data[col].min()), int(data[col].max())) for col in ["backlog_at_open", "staff_on_shift"]},
    }


def predict_wait(bundle, inputs):
    """(low, middle, high) predicted wait in hours for one request described by `inputs`."""
    row = prepare_features(pd.DataFrame([inputs]), bundle["categories"])
    low, mid, high = sorted(float(bundle["models"][q].predict(row)[0]) for q in QUANTILES)
    return max(low, 0.0), max(mid, 0.0), max(high, 0.0)


def importance_chart(bundle):
    """Horizontal bars of permutation importance, the strongest driver on top in HIGHLIGHT."""
    imp = bundle["importance"].clip(lower=0).sort_values()      # negative = noise, shown as 0
    labels = [FEATURE_NAMES[f] for f in imp.index]
    colors = [HIGHLIGHT if f == imp.index[-1] else GREY for f in imp.index]
    fig, ax = plt.subplots(figsize=(7, 3.5))
    ax.barh(labels, imp.values, color=colors)
    ax.set_xlim(left=0)
    ax.set_xlabel("extra error in hours when the input is shuffled")
    ax.set_title(f"{FEATURE_NAMES[imp.index[-1]]} drives the predicted wait most")
    plt.close(fig)
    return fig


def unusual_requests(bundle, df, top=10):
    """Requests that waited much longer than the model expected for their conditions."""
    pred = bundle["oof_pred"].dropna()
    table = df.loc[pred.index, ["request_id", "opened_at", "desk", "priority", "request_type",
                                "backlog_at_open", "staff_on_shift", "wait_hours"]].copy()
    table["expected_wait"] = pred.round(2)
    table["extra_hours"] = (table["wait_hours"] - table["expected_wait"]).round(2)
    return table.sort_values("extra_hours", ascending=False).head(top).reset_index(drop=True)


# ---------------------------------------------------------------------------
# Training animation: watch the model learn, tree by tree
# ---------------------------------------------------------------------------
ANIM_TREES = 1000          # trained past our choice of 300 so overfitting becomes visible
CHOSEN_TREES = 300
ANIM_STAGES = [0, 1, 2, 3, 5, 7, 10, 14, 20, 27, 35, 45, 60, 75, 90, 110, 130, 160, 200,
               250, 300, 375, 450, 550, 650, 800, 1000]


def learning_curves(df, seed=0):
    """Everything the animation needs, from the same 75/25 split as the scorecard. None if too few rows."""
    data = df.dropna(subset=FEATURES + ["wait_hours"])
    if len(data) < MIN_ROWS:
        return None
    categories = {col: sorted(data[col].unique()) for col in CATEGORICAL}
    X, y = prepare_features(data, categories), data["wait_hours"]
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.25, random_state=seed)

    models = {}
    for q in QUANTILES:
        m = _model(q, seed)
        m.set_params(max_iter=ANIM_TREES)
        models[q] = m.fit(X_tr, y_tr)

    # error after every tree (stage 0 = the starting guess: the median of the training waits)
    start_tr = np.full(len(y_tr), y_tr.median())
    start_te = np.full(len(y_te), y_tr.median())
    train_err = [mean_absolute_error(y_tr, start_tr)] + [mean_absolute_error(y_tr, p) for p in models[0.5].staged_predict(X_tr)]
    test_err = [mean_absolute_error(y_te, start_te)] + [mean_absolute_error(y_te, p) for p in models[0.5].staged_predict(X_te)]

    # average prediction at each backlog value (partial dependence), after each stage
    grid = np.arange(int(X_tr["backlog_at_open"].min()), int(X_tr["backlog_at_open"].max()) + 1)
    X_grid = pd.concat([X_tr.assign(backlog_at_open=g) for g in grid], ignore_index=True)
    wanted = set(ANIM_STAGES)
    curves = {}
    for q, m in models.items():
        stage_curves = {0: np.full(len(grid), np.quantile(y_tr, q))}
        for i, p in enumerate(m.staged_predict(X_grid), start=1):
            if i in wanted:
                stage_curves[i] = p.reshape(len(grid), len(X_tr)).mean(axis=1)
        curves[q] = stage_curves

    return {"grid": grid, "curves": curves, "train_err": np.array(train_err), "test_err": np.array(test_err),
            "points": data[["backlog_at_open", "wait_hours"]], "best_tree": int(np.argmin(test_err))}


def learning_animation(curves):
    """Plotly figure with a Play button: left, the model's curve forming; right, train/test error per tree."""
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    grid, c = curves["grid"], curves["curves"]
    tr_err, te_err = curves["train_err"], curves["test_err"]
    baseline = te_err[0]
    pts = curves["points"]
    y_top = float(pts["wait_hours"].quantile(0.99)) * 1.1
    jitter = np.random.default_rng(0).uniform(-0.3, 0.3, len(pts))

    fig = make_subplots(rows=1, cols=2, column_widths=[0.55, 0.45], horizontal_spacing=0.1,
                        subplot_titles=("What the model has learned", "Error after each tree"))

    def moving_traces(k):
        return [
            go.Scatter(x=grid, y=c[0.9][k], mode="lines", line=dict(width=0), hoverinfo="skip", showlegend=False),
            go.Scatter(x=grid, y=c[0.1][k], mode="lines", line=dict(width=0), fill="tonexty",
                       fillcolor="rgba(79,41,183,0.18)", name="10%-90% range", hoverinfo="skip"),
            go.Scatter(x=grid, y=c[0.5][k], mode="lines", line=dict(color=PURPLE, width=4), name="model prediction",
                       hovertemplate="backlog %{x}: %{y:.1f} h<extra></extra>"),
            go.Scatter(x=np.arange(k + 1), y=tr_err[:k + 1], mode="lines", line=dict(color=GREY, width=2),
                       name="error on training data"),
            go.Scatter(x=np.arange(k + 1), y=te_err[:k + 1], mode="lines", line=dict(color=HIGHLIGHT, width=3),
                       name="error on unseen data"),
            go.Scatter(x=[k], y=[te_err[k]], mode="markers", marker=dict(color=HIGHLIGHT, size=12),
                       showlegend=False, hovertemplate="tree %{x}: %{y:.2f} h<extra></extra>"),
        ]

    # static traces first: the real requests (left) and the simple-guess line (right)
    fig.add_trace(go.Scatter(x=pts["backlog_at_open"] + jitter, y=pts["wait_hours"], mode="markers",
                             marker=dict(color=GREY, size=5, opacity=0.45), name="real requests",
                             hoverinfo="skip"), row=1, col=1)
    fig.add_trace(go.Scatter(x=[0, ANIM_TREES], y=[baseline, baseline], mode="lines",
                             line=dict(color="black", dash="dash", width=1.5), name="simple guess (median)"),
                  row=1, col=2)
    first = moving_traces(0)
    for i, tr in enumerate(first):
        fig.add_trace(tr, row=1, col=1 if i < 3 else 2)
    moving_ids = list(range(2, 2 + len(first)))

    def title(k):
        if k == 0:
            return "Tree 0: the model starts from the simple guess (the median wait)"
        note = "  (our model)" if k == CHOSEN_TREES else ""
        return (f"Tree {k}: error on unseen requests {te_err[k]:.2f} h vs simple guess {baseline:.2f} h"
                f"  ({1 - te_err[k] / baseline:.0%} better){note}")

    fig.frames = [go.Frame(name=str(k), data=moving_traces(k), traces=moving_ids,
                           layout=go.Layout(title_text=title(k))) for k in ANIM_STAGES]

    fig.add_vline(x=CHOSEN_TREES, line=dict(color=PURPLE, dash="dot"), row=1, col=2)
    fig.add_annotation(x=CHOSEN_TREES, y=baseline, yshift=10, text=f"our model: {CHOSEN_TREES} trees",
                       showarrow=False, font=dict(color=PURPLE), xanchor="left", row=1, col=2)

    play = dict(label="▶ Play", method="animate",
                args=[None, dict(frame=dict(duration=350, redraw=True), transition=dict(duration=0),
                                 fromcurrent=True, mode="immediate")])
    pause = dict(label="❚❚ Pause", method="animate",
                 args=[[None], dict(frame=dict(duration=0, redraw=False), mode="immediate")])
    steps = [dict(label=str(k), method="animate",
                  args=[[str(k)], dict(frame=dict(duration=0, redraw=True), mode="immediate")]) for k in ANIM_STAGES]

    fig.update_layout(
        title=dict(text=title(0), x=0, xanchor="left", y=0.97, yanchor="top"),
        height=620, template="plotly_white", margin=dict(t=110, b=150, l=60, r=30),
        legend=dict(orientation="h", x=0, y=1.13, yanchor="bottom"),
        updatemenus=[dict(type="buttons", direction="left", x=0, y=-0.2, xanchor="left", yanchor="top",
                          showactive=False, buttons=[play, pause])],
        sliders=[dict(active=0, steps=steps, x=0.17, y=-0.2, len=0.83, yanchor="top",
                      currentvalue=dict(prefix="trees: ", visible=True, xanchor="left"), pad=dict(t=0, b=0))],
    )
    fig.update_xaxes(title_text="backlog at open (requests waiting)", range=[grid.min() - 1, grid.max() + 1], row=1, col=1)
    fig.update_yaxes(title_text="wait (hours)", range=[0, y_top], row=1, col=1)
    fig.update_xaxes(title_text="number of trees", range=[0, ANIM_TREES], row=1, col=2)
    fig.update_yaxes(title_text="average error (hours)", range=[min(tr_err) * 0.9, baseline * 1.08], row=1, col=2)
    return fig
