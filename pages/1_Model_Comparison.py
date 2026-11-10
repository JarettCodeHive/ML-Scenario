"""Model Comparison dashboard: which model fits the Cluster C bookings target best?

Streamlit picks this file up automatically as a second page of app.py.
It reads data/models/model_comparison/ (written by scripts/compare_models.py)
and can re-run the comparison from the page.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from cluster_c.model_comparison import (  # noqa: E402
    BASELINE_MODEL,
    COMPARISON_DIR,
    CURRENT_MODEL,
    MONOTONIC_THRESHOLD_PCT,
    different_models,
    load_comparison,
    save_comparison,
)
from cluster_c.preparation import TARGET  # noqa: E402

st.set_page_config(page_title="Model Comparison · Cluster C", page_icon="📊", layout="wide")

# Categorical slots (fixed order): 1 blue, 2 orange, 3 aqua. Others are muted gray.
BLUE, ORANGE, AQUA, GRAY = "#2a78d6", "#eb6834", "#1baf7a", "#9a9a94"
TERM_COLORS = {"1 Year": BLUE, "3 Year": ORANGE, "5 Year": AQUA}
FEATURE_PATH = ROOT / "data/prepared/term_model_features.parquet"


def bar_colors(models: pd.Series, recommended: str) -> list[str]:
    return [ORANGE if m == CURRENT_MODEL else BLUE if m == recommended else GRAY for m in models]


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(height=height, margin=dict(l=10, r=60, t=50, b=10), legend_title_text="",
                      bargap=0.35, hoverlabel=dict(namelength=-1))
    fig.update_xaxes(showgrid=True, gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    fig.update_yaxes(showgrid=False)
    return fig


@st.cache_data
def load_results(stamp: float):
    return load_comparison(COMPARISON_DIR)


st.title("Model comparison — Cluster C bookings model")
st.caption("Which model should predict bookings for the term-flexibility what-if? "
           "All models use the same chronological split, features and test period as the production CatBoost.")

with st.sidebar:
    st.subheader("Re-run comparison")
    st.caption("Trains every model again on data/prepared/term_model_features.parquet (about 1–2 minutes).")
    folds = st.number_input("Backtest folds", 0, 6, 3, help="Expanding-window folds inside train + validation. 0 = skip.")
    if st.button("Run comparison now", type="primary"):
        frame = pd.read_parquet(FEATURE_PATH, engine="pyarrow")
        with st.status("Training models…", expanded=True) as status:
            result = different_models(frame, n_backtest_folds=int(folds), progress=status.write)
            save_comparison(result)
            status.update(label="Comparison finished", state="complete")
        st.cache_data.clear()

metadata_path = COMPARISON_DIR / "metadata.json"
if not metadata_path.exists():
    st.info("No comparison results yet. Click **Run comparison now** in the sidebar, or run:")
    st.code("PYTHONPATH=src python scripts/compare_models.py", language="bash")
    st.stop()

res = load_results(metadata_path.stat().st_mtime)
board, meta, scenario = res.leaderboard, res.metadata, res.scenario
recommended = meta["recommended_model"]
best = board.sort_values("test_wape_pct").iloc[0]
current = board[board.model == CURRENT_MODEL]
base = board[board.model == BASELINE_MODEL]

# ---------------------------------------------------------------- target
with st.container(border=True):
    t1, t2 = st.columns([3, 2])
    t1.markdown(
        f"#### 🎯 Target variable: `{TARGET}`\n"
        "**Total committed bookings for the whole term, in USD.** In this data it equals "
        "`ANNUALIZED_NET_PRICE_USD_AMT × term years` exactly, so a 5-year deal books roughly 5× its annual value. "
        "That is why `TERM_BUCKET` is such a strong feature, and why one variant below learns the **annual** value "
        "and multiplies back by the term."
    )
    t2.markdown(
        f"**Split** — train {meta['split_rows']['train']:,} · validation {meta['split_rows']['validation']:,} · "
        f"test {meta['split_rows']['test']:,} rows  \n"
        f"Test period: {meta['split_dates']['test'][0]} → {meta['split_dates']['test'][1]}  \n"
        f"Dataset: `{meta['source_dataset_version']}`  \n"
        f"Run: {meta['created_at_utc'][:16].replace('T', ' ')} UTC"
    )

# ---------------------------------------------------------------- KPIs
k1, k2, k3, k4 = st.columns(4)
with k1.container(border=True):
    st.caption("Recommended model")
    st.markdown(f"### {recommended}")
with k2.container(border=True):
    st.caption("Most accurate on test")
    st.markdown(f"### {best.model}")
    st.caption(f"Test WAPE {best.test_wape_pct:.2f}%")
if not current.empty:
    c = current.iloc[0]
    k3.metric("Current CatBoost test WAPE", f"{c.test_wape_pct:.2f}%",
              f"{c.test_wape_pct - best.test_wape_pct:+.2f} pts vs best", delta_color="inverse")
if not base.empty:
    k4.metric("Simple median baseline WAPE", f"{base.iloc[0].test_wape_pct:.2f}%",
              help="How wrong you are with no ML at all: median bookings by term and suite.")
st.info(f"**Why:** {meta['recommendation_reason']}")

# ---------------------------------------------------------------- leaderboard
st.subheader("1 · Leaderboard (sorted by test WAPE — lower is better)")
table = board[[
    "rank", "model", "family", "test_wape_pct", "test_mae", "test_rmse", "test_r2", "test_mape_pct",
    "test_within_20pct", "test_bias_pct", "backtest_wape_mean", "backtest_wape_std", "monotonic_pct",
    "default_scenario_change_pct", "fit_seconds",
]].rename(columns={
    "test_wape_pct": "WAPE %", "test_mae": "MAE $", "test_rmse": "RMSE $", "test_r2": "R²", "test_mape_pct": "MAPE %",
    "test_within_20pct": "Within ±20 % of deals", "test_bias_pct": "Bias %", "backtest_wape_mean": "Backtest WAPE %",
    "backtest_wape_std": "Backtest spread ±", "monotonic_pct": "1y<3y<5y order %",
    "default_scenario_change_pct": "Default what-if %", "fit_seconds": "Fit sec",
})
st.dataframe(
    table, hide_index=True, width="stretch",
    column_config={
        "MAE $": st.column_config.NumberColumn(format="$%,.0f"), "RMSE $": st.column_config.NumberColumn(format="$%,.0f"),
        **{col: st.column_config.NumberColumn(format="%.2f") for col in
           ["WAPE %", "R²", "MAPE %", "Within ±20 % of deals", "Bias %", "Backtest WAPE %", "Backtest spread ±",
            "1y<3y<5y order %", "Default what-if %", "Fit sec"]},
    },
)
st.caption("Orange = current CatBoost · Blue = recommended (if different) · Gray = other models. "
           "Default what-if = commitment change for 100% flexible deals, 50% five-year, vs all 3-year.")

# ---------------------------------------------------------------- accuracy + stability
col_a, col_b = st.columns(2)
ordered = board.sort_values("test_wape_pct", ascending=False)
fig = go.Figure(go.Bar(
    x=ordered.test_wape_pct, y=ordered.model, orientation="h",
    marker=dict(color=bar_colors(ordered.model, recommended), cornerradius=4),
    text=[f"{v:.1f}%" for v in ordered.test_wape_pct], textposition="outside", cliponaxis=False,
    hovertemplate="%{y}<br>Test WAPE %{x:.2f}%<extra></extra>",
))
fig.update_layout(title="Test WAPE by model", xaxis_title="WAPE % (lower is better)")
col_a.plotly_chart(style_fig(fig), width="stretch")

if not res.backtest.empty:
    bt = board.dropna(subset=["backtest_wape_mean"]).sort_values("backtest_wape_mean", ascending=False)
    fig = go.Figure(go.Scatter(
        x=bt.backtest_wape_mean, y=bt.model, mode="markers",
        marker=dict(size=11, color=bar_colors(bt.model, recommended), line=dict(width=2, color="white")),
        error_x=dict(type="data", array=bt.backtest_wape_std, color="rgba(128,128,128,0.6)", thickness=2, width=6),
        hovertemplate="%{y}<br>Backtest WAPE %{x:.2f}% ± %{error_x.array:.2f}<extra></extra>",
    ))
    fig.update_layout(title=f"Stability: backtest WAPE mean ± spread ({meta['backtest_folds']} time folds)",
                      xaxis_title="WAPE % (lower and tighter is better)")
    col_b.plotly_chart(style_fig(fig), width="stretch")
else:
    col_b.info("Backtest was skipped (0 folds).")

# ---------------------------------------------------------------- per term
st.subheader("2 · Accuracy by term bucket")
terms = res.metrics_by_term[res.metrics_by_term.split == "test"].copy()
terms["model"] = pd.Categorical(terms.model, categories=board.model.tolist(), ordered=True)
fig = px.bar(terms.sort_values("model"), x="model", y="wape_pct", color="term_bucket", barmode="group",
             color_discrete_map=TERM_COLORS, labels={"wape_pct": "Test WAPE %", "model": "", "term_bucket": "Term"},
             title="Test WAPE per term — a good model is accurate for 1-, 3- and 5-year deals alike",
             hover_data={"rows": True, "mae": ":,.0f"})
fig.update_traces(marker_cornerradius=4, marker_line_width=0)
fig.update_layout(bargroupgap=0.08)
st.plotly_chart(style_fig(fig, 400), width="stretch")

# ---------------------------------------------------------------- scenario fitness
st.subheader("3 · Scenario fitness — does the model react sensibly when the term changes?")
st.markdown(
    "The what-if engine re-scores every deal as 1, 3 and 5 years. Two models can have the same WAPE "
    "but give **very different what-if answers**. This section checks that: "
    f"a model is *scenario-safe* when at least {MONOTONIC_THRESHOLD_PCT:.0f}% of test deals keep the order "
    "1-year < 3-year < 5-year bookings."
)
s1, s2 = st.columns(2)
sc = scenario.set_index("model").loc[board.model].reset_index().sort_values("default_scenario_change_pct")
fig = go.Figure(go.Bar(
    x=sc.default_scenario_change_pct, y=sc.model, orientation="h",
    marker=dict(color=bar_colors(sc.model, recommended), cornerradius=4),
    text=[f"{v:+.1f}%" for v in sc.default_scenario_change_pct], textposition="outside", cliponaxis=False,
    hovertemplate="%{y}<br>Commitment change %{x:+.2f}%<extra></extra>",
))
truth = meta.get("synthetic_true_default_change_pct")
if truth is not None:
    fig.add_vline(x=truth, line=dict(color="rgba(80,80,80,0.55)", width=1.5, dash="dash"),
                  annotation_text=f"Known true effect in synthetic data: {truth:+.2f}%", annotation_position="top")
fig.update_layout(title="Default what-if answer by model (100% flexible, 50% five-year)",
                  xaxis_title="Total commitment change vs all-3-year (%)")
s1.plotly_chart(style_fig(fig), width="stretch")

ratio = sc.melt(id_vars="model", value_vars=["annual_ratio_1y_vs_3y", "annual_ratio_5y_vs_3y"],
                var_name="term", value_name="ratio")
ratio["term"] = ratio.term.map({"annual_ratio_1y_vs_3y": "1 Year", "annual_ratio_5y_vs_3y": "5 Year"})
fig = px.scatter(ratio, x="ratio", y="model", color="term", color_discrete_map=TERM_COLORS,
                 labels={"ratio": "Predicted annual value ÷ 3-year annual value", "model": "", "term": "Term"},
                 title="Implied annual-value effect of term (1.00 = same as 3-year)")
fig.update_traces(marker=dict(size=11, line=dict(width=2, color="white")))
obs = scenario.iloc[0]
fig.add_vline(x=1.0, line=dict(color="rgba(128,128,128,0.5)", width=1))
fig.add_vline(x=obs.observed_ratio_5y_vs_3y, line=dict(color=AQUA, width=1, dash="dot"),
              annotation_text="raw data 5y", annotation_position="bottom right")
fig.add_vline(x=obs.observed_ratio_1y_vs_3y, line=dict(color=BLUE, width=1, dash="dot"),
              annotation_text="raw data 1y", annotation_position="bottom left")
s2.plotly_chart(style_fig(fig), width="stretch")
st.caption(
    "Dotted lines show the raw medians in the training data. Raw medians are **confounded** (larger customers pick "
    "5-year terms more often), so a good model should land between 1.00 and the dotted line, not on it."
    + (" On synthetic data the dashed line is the true effect, so the closest bar is the most trustworthy what-if."
       if truth is not None else "")
)
scenario_cols = ["model", "monotonic_pct", "annual_ratio_1y_vs_3y", "annual_ratio_5y_vs_3y", "default_scenario_change_pct"]
if "scenario_error_vs_truth_pts" in scenario:
    scenario_cols.append("scenario_error_vs_truth_pts")
st.dataframe(scenario[scenario_cols].round(3), hide_index=True, width="stretch")

# ---------------------------------------------------------------- three lenses
st.subheader("4 · The three business lenses, per model")
st.markdown(
    "The same Cluster C maths the main page uses (`scenario.py`), run on each model's 1/3/5-year scores at the main page's "
    "default sliders (100% flexible, 50% five-year, 8% discount rate), on the test deals."
)
lenses = res.lenses
if lenses.empty:
    st.info("Re-run the comparison to calculate the lens check.")
else:
    from cluster_c.model_comparison import SYNTHETIC_TRUTH_LABEL  # noqa: E402

    view = lenses.rename(columns={
        "growth_commitment_change_pct": "Growth · commitment change %", "growth_npv_change_pct": "Growth · NPV change %",
        "growth_npv_breakeven_5y_share_pct": "Growth · NPV breakeven 5y share %",
        "customer_avg_term_years": "Customer value · avg term (yrs)",
        "simplicity_coverage_cv_pct": "Simplicity · coverage volatility %",
        "simplicity_renewal_points_change": "Simplicity · renewal points Δ",
        "growth_commitment_change_error_vs_truth_pts": "Commitment error vs truth (pts)",
        "growth_npv_change_error_vs_truth_pts": "NPV error vs truth (pts)",
        "simplicity_coverage_cv_error_vs_truth_pts": "Volatility error vs truth (pts)",
    }).drop(columns=["simplicity_coverage_cv_change_pts"], errors="ignore")
    st.dataframe(view.round(2), hide_index=True, width="stretch")
    l1, l2 = st.columns(2)
    models_only = lenses[lenses.model != SYNTHETIC_TRUTH_LABEL].sort_values("growth_npv_change_pct")
    truth_row = lenses[lenses.model == SYNTHETIC_TRUTH_LABEL]
    for column, title, target in (
        ("growth_npv_change_pct", "Growth lens: NPV change vs all-3-year (%)", l1),
        ("simplicity_coverage_cv_pct", "Simplicity lens: 5-year coverage volatility (%)", l2),
    ):
        ordered_l = models_only.sort_values(column)
        # Dot plot: values cluster tightly and can be negative, so bars from 0 would hide the differences.
        fig = go.Figure(go.Scatter(
            x=ordered_l[column], y=ordered_l.model, mode="markers+text",
            marker=dict(size=12, color=bar_colors(ordered_l.model, recommended), line=dict(width=2, color="white")),
            text=[f"{v:+.2f}%" if column.startswith("growth") else f"{v:.2f}%" for v in ordered_l[column]],
            textposition="middle right", cliponaxis=False, hovertemplate="%{y}<br>%{x:.2f}%<extra></extra>",
        ))
        if column.startswith("growth"):
            fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.5)", width=1))
        if not truth_row.empty:
            fig.add_vline(x=float(truth_row.iloc[0][column]), line=dict(color="rgba(80,80,80,0.55)", width=1.5, dash="dash"),
                          annotation_text=f"Known truth {float(truth_row.iloc[0][column]):.2f}%", annotation_position="top")
        fig.update_layout(title=title, xaxis_title="%")
        target.plotly_chart(style_fig(fig), width="stretch")
    st.caption(
        "**Customer value** and **renewal decision points** come straight from the sliders in the current app "
        "(they do not use the model), so they are identical for every model. Measuring customer value properly needs "
        "renewal, churn and acceptance data (see DATA_CONTRACT.md). The model choice moves the **Growth** and "
        "**Simplicity (volatility)** lenses — check the sign of the NPV change in particular."
    )

# ---------------------------------------------------------------- drill-down
st.subheader("5 · Drill into one model")
default_index = board.model.tolist().index(recommended) if recommended in board.model.tolist() else 0
chosen = st.selectbox("Model", board.model.tolist(), index=default_index)
pred = res.predictions[res.predictions.model == chosen].copy()
pred["error_pct"] = (pred.predicted - pred.actual) / pred.actual * 100
d1, d2 = st.columns(2)
top = max(pred.actual.max(), pred.predicted.max())
fig = px.scatter(pred, x="actual", y="predicted", color="TERM_BUCKET", color_discrete_map=TERM_COLORS, opacity=0.65,
                 labels={"actual": "Actual bookings (USD)", "predicted": "Predicted bookings (USD)", "TERM_BUCKET": "Term"},
                 hover_data={"SUBSCRIPTION_REFERENCE_ID": True, "SUITE_NAME": True, "actual": ":,.0f", "predicted": ":,.0f"},
                 title=f"{chosen}: predicted vs actual (test deals)")
fig.add_trace(go.Scatter(x=[0, top], y=[0, top], mode="lines", line=dict(color="rgba(128,128,128,0.6)", dash="dash"),
                         name="Perfect prediction", hoverinfo="skip"))
fig.update_traces(marker=dict(size=8), selector=dict(mode="markers"))
d1.plotly_chart(style_fig(fig, 420), width="stretch")
fig = px.histogram(pred, x=pred.error_pct.clip(-100, 100), nbins=50, color_discrete_sequence=[BLUE],
                   labels={"x": "Prediction error % (clipped to ±100)"}, title=f"{chosen}: error distribution")
fig.add_vline(x=0, line=dict(color="rgba(128,128,128,0.7)", width=1))
fig.update_traces(marker_line_width=1, marker_line_color="white", hovertemplate="Error %{x}<br>%{y} deals<extra></extra>")
fig.update_layout(yaxis_title="Deals")
d2.plotly_chart(style_fig(fig, 420), width="stretch")

seg = pred.groupby("SUITE_NAME").apply(
    lambda g: pd.Series({"deals": len(g), "WAPE %": (g.predicted - g.actual).abs().sum() / g.actual.abs().sum() * 100,
                         "Bias %": (g.predicted - g.actual).sum() / g.actual.sum() * 100}),
    include_groups=False,
).reset_index()
st.dataframe(seg.round(2), hide_index=True, width="stretch")

# ---------------------------------------------------------------- glossary
with st.expander("What each metric means"):
    st.markdown(
        """
- **WAPE %** (primary) — total absolute error ÷ total actual bookings. "We are off by X% of the money." Large deals count more, which matches a portfolio what-if.
- **MAE $ / RMSE $** — average error per deal in dollars; RMSE punishes big misses more.
- **R²** — share of the variation in bookings the model explains (1.0 = perfect).
- **MAPE %** — average per-deal % error; small deals weigh as much as large ones.
- **Within ±20 % of deals** — share of test deals predicted within 20% of actual.
- **Bias %** — negative = the model under-predicts total bookings overall.
- **Backtest WAPE % / spread** — WAPE averaged over earlier time folds, and how much it moves; a gap between two models smaller than the spread is noise.
- **1y<3y<5y order %** — share of deals where the model keeps the natural order of term bookings when the term is switched.
- **Default what-if %** — the answer the scenario engine would give with this model at 100% flexible adoption and 50% five-year share.
"""
    )
with st.expander("Model descriptions"):
    st.dataframe(board[["model", "family", "description"]], hide_index=True, width="stretch")
