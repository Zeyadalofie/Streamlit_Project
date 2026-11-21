# Police Requests Dashboard

A tested Streamlit dashboard for one quarter of police service requests, plus an AI page that predicts waiting times and shows the model learning.

> All data are synthetic police administrative records. The dashboard demonstrates a method; it is not an operational tool.

Team project, Week 07 (Notebook 10), Tuwaiq Academy AI/ML Bootcamp.

---

## What it does

### Page 1: Dashboard (`app.py`)

Supervisors filter the requests and read the results.

- **Sidebar filters:** Region, Channel, Priority, Dates, and a **Reset filters** button.
- **Five indicators:** Requests, Median minutes, Open share, Median wait, Survey answered.
- **Four tabs:**
  - **Overview:** requests per complete week. Partial weeks are left out so they don't show a false dip.
  - **Desks:** one row per desk, plus a bar chart of median wait with the slowest desk highlighted.
  - **Findings:** the slowest and fastest desk by median wait, with a bootstrap 95% interval and an evidence label (`weak`, `moderate` or `strong`).
  - **Data:** the filtered rows and a **Download filtered CSV** button.
- **Method and data notes:** the date range of the file and the survey response rate.

Every part of the page handles an **empty selection**. If the filters match nothing, the page shows a warning and `"0"` / `"—"` instead of crashing or showing wrong numbers.

### Page 2: AI Wait Predictor (`pages/ai_predictor.py`)

A gradient boosting model trained on the whole quarter.

1. **Watch the model learn:** an animation (press Play) of the model being built tree by tree. The left panel shows the prediction curve forming over the real requests. The right panel shows the error on training data and on unseen data, which makes the overfitting visible.
2. **Is the model better than a simple guess?** Error on requests the model never saw, compared with predicting the median for every request.
3. **Predict the wait for a request:** enter the backlog, staff on shift, priority, request type and desk to get a predicted wait with an 80% range.
4. **What drives the prediction?** Permutation importance for each input.
5. **Requests that waited much longer than expected:** requests whose wait was far above what the model expected for their conditions, using cross-validated predictions.

---

## Key results (full data, 600 requests)

| Fact | Value |
| --- | --- |
| Requests | 600 |
| Busiest desk | Central (142 requests) |
| Finding | North had the longest median wait (3.5 h) and West the shortest (2.7 h). |
| Evidence | **strong**: 95% interval for the difference 0.11 to 1.44 h, smaller desk n = 72 |
| Model error (unseen requests) | 1.08 h, vs 1.68 h for the simple guess (**36% better**) |
| 80% range hit rate | 71% (the range is slightly too narrow) |
| Strongest driver | Backlog at open, then staff on shift |

---

## Project structure

```text
team_project/
├── README.md
├── requirements.txt
├── team_app/
│   ├── app.py              # Page 1: the dashboard (team)
│   ├── dash_data.py        # Role A: load_requests, filter_requests
│   ├── dash_metrics.py     # Role B: compute_kpis, desk_table
│   ├── dash_charts.py      # Role C: weekly_chart, desk_chart
│   ├── dash_findings.py    # Role D: wait_finding, data_notes
│   ├── dash_tools.py       # provided: week_table, diff_ci, evidence_label, colours
│   ├── dash_ai.py          # AI extension: model, predictions, importance, animation
│   └── pages/
│       └── ai_predictor.py # Page 2: the AI Wait Predictor
└── team_files/             # test CSV files written by the notebook
```

The data are read directly from the course repository:
`https://raw.githubusercontent.com/hanyelshafey/ai-security-bootcamp-data/main/week_07/police_requests_team.csv`

---

## How to run

Requires Python 3.10 or newer and an internet connection (for the data file).

From the `team_project` folder (the parent of `team_app`):

```bash
python -m pip install -r requirements.txt
python -m streamlit run team_app/app.py
```

The dashboard opens at `http://localhost:8501`. The AI page is in the sidebar under **ai predictor**. The first visit trains the models (about 10 seconds); after that the page is instant.

Use `python -m` in both commands so the libraries are installed into the same Python that runs Streamlit.

---

## Team and contributions

| Member | Role | Files |
| --- | --- | --- |
| Omar Al-Kaibari | A - Data | `dash_data.py`, `app.py` |
| Zeyad Al-Harbi | B - Indicators | `dash_metrics.py`, `app.py` |
| Naife Al-metarie | C - Charts | `dash_charts.py`, `app.py` |
| Naife Al-anize | D - Findings | `dash_findings.py`, `app.py` |

Each role owns one module with a fixed interface contract, so the four modules could be written in parallel and integrated in `app.py`.

---

## How we know the numbers are correct

- Every module function is tested on the full table **and on an empty table**.
- The dashboard is run headlessly with Streamlit's `AppTest` on four states: all data, no region selected (empty view), after **Reset filters**, and a 3-day date range (no complete week).
- The data notes are tested on a filtered view, where bugs that use the whole file instead of the view become visible.
- The AI page is tested with the full file and with a 40-row file (it must show a warning instead of training a weak model).
- Every AI score is measured on data the model did not see: a 25% test split for the scorecard and 5-fold cross-validation for the unusual-wait table.

---

## Limitations

- **One quarter of synthetic data.** Real records would need checking for missing or wrong values, and the model would need retraining over time.
- **Correlation is not causation.** The model links more staff with shorter waits, but it cannot say what would happen if staffing changed. The dashboard gives no staffing recommendation.
- **Small sample.** 600 requests limit the model's accuracy; the trees are kept shallow (depth 3, 300 trees) to avoid overfitting.
- **The 80% range is slightly too narrow** (71% hit rate on unseen data).

## Possible improvements

- Calibrate the prediction range (for example with conformal prediction) so it reaches the 80% target.
- Add more quarters of data and monitor the model's error over time.

---

## Notes

- **Sliders and Arabic browsers:** Streamlit sliders run backwards when the browser language is Arabic, so the AI page uses number boxes (− / +) instead.
- **Notebook cells:** the notebook writes `app.py` and the role modules with `%%writefile`. Keep those cells identical to the files here, or **Restart and Run All** will overwrite the files.
