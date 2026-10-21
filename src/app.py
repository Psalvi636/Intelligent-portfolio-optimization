from pathlib import Path
import json

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
DATA_DIR = PROJECT_ROOT / "data"

FUZZY_WEEKLY = RESULTS_DIR / "fuzzy_ga_test_final_with_weights.csv"
FUZZY_METRICS = RESULTS_DIR / "fuzzy_ga_test_final_with_weights_metrics.json"
PPO_WEEKLY = RESULTS_DIR / "ppo_test_final_weekly.csv"
PPO_METRICS = RESULTS_DIR / "ppo_test_final_metrics.json"

STOCKS = [
    "RELIANCE",
    "HDFCBANK",
    "TCS",
    "HINDUNILVR",
    "SUNPHARMA",
    "LT",
]

st.set_page_config(
    page_title="RL vs Fuzzy-GA | Portfolio Optimization",
    page_icon="📈",
    layout="wide",
)

st.markdown(
    """
    <style>
    .main {
        padding-top: 1rem;
    }

    .hero {
        padding: 1.5rem 1.8rem;
        border-radius: 14px;
        margin-bottom: 1.2rem;
        background: linear-gradient(
            135deg,
            rgba(30, 41, 59, 0.95),
            rgba(15, 23, 42, 0.95)
        );
        color: white;
    }

    .hero h1 {
        margin: 0;
        font-size: 2.2rem;
    }

    .hero p {
        margin: 0.45rem 0 0 0;
        font-size: 1.02rem;
        opacity: 0.82;
    }

    .metric-card {
        padding: 1rem;
        border: 1px solid rgba(128, 128, 128, 0.25);
        border-radius: 12px;
        min-height: 105px;
    }

    .metric-label {
        font-size: 0.82rem;
        opacity: 0.7;
        margin-bottom: 0.3rem;
    }

    .metric-value {
        font-size: 1.55rem;
        font-weight: 700;
    }

    .winner {
        font-weight: 700;
    }

    .section-note {
        opacity: 0.7;
        font-size: 0.88rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

def load_json(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)

def load_results():
    required = [
        FUZZY_WEEKLY,
        FUZZY_METRICS,
        PPO_WEEKLY,
        PPO_METRICS,
    ]

    missing = [str(p) for p in required if not p.exists()]

    if missing:
        raise FileNotFoundError(
            "The following required result files are missing:\n\n"
            + "\n".join(missing)
        )

    fuzzy = pd.read_csv(FUZZY_WEEKLY)
    ppo = pd.read_csv(PPO_WEEKLY)

    fuzzy["decision_date"] = pd.to_datetime(fuzzy["decision_date"])
    fuzzy["holding_start"] = pd.to_datetime(fuzzy["holding_start"])
    fuzzy["holding_end"] = pd.to_datetime(fuzzy["holding_end"])

    ppo["Observation_Date"] = pd.to_datetime(ppo["Observation_Date"])
    ppo["Holding_Start"] = pd.to_datetime(ppo["Holding_Start"])
    ppo["Holding_End"] = pd.to_datetime(ppo["Holding_End"])

    fuzzy_metrics = load_json(FUZZY_METRICS)
    ppo_metrics = load_json(PPO_METRICS)

    return fuzzy, ppo, fuzzy_metrics, ppo_metrics

def normalize_model_data(fuzzy, ppo):
    """Create a common schema without modifying source files."""

    f = pd.DataFrame({
        "date": fuzzy["holding_end"],
        "decision_date": fuzzy["decision_date"],
        "return": fuzzy["weekly_return"],
        "turnover": fuzzy["turnover"],
        "transaction_cost": fuzzy["transaction_cost"],
        "portfolio_value": fuzzy["portfolio_value"],
        "model": "Fuzzy-GA",
    })

    p = pd.DataFrame({
        "date": ppo["Holding_End"],
        "decision_date": ppo["Observation_Date"],
        "return": ppo["PPO_Net_Return"],
        "turnover": ppo["PPO_Turnover"],
        "transaction_cost": ppo["PPO_Transaction_Cost"],
        "model": "PPO",
    })

    # PPO does not need a supplied portfolio-value column:
    # reconstruct it from net weekly returns.
    p["portfolio_value"] = (1.0 + p["return"]).cumprod()

    return f, p

def metric_value(metrics, key, default=np.nan):
    """
    Read a metric from either the Fuzzy-GA or PPO metrics JSON.

    Fuzzy-GA stores performance metrics at the top level.

    PPO stores performance metrics inside:
        metrics["ppo"]

    PPO turnover and transaction costs are stored at the top level
    with PPO-specific names.
    """

    # --------------------------------------------------------
    # Direct/top-level metric
    # --------------------------------------------------------
    if key in metrics:
        value = metrics[key]

        try:
            return float(value)
        except (TypeError, ValueError):
            pass

    # --------------------------------------------------------
    # PPO nested performance metrics
    # --------------------------------------------------------
    if "ppo" in metrics and isinstance(metrics["ppo"], dict):
        if key in metrics["ppo"]:
            value = metrics["ppo"][key]

            try:
                return float(value)
            except (TypeError, ValueError):
                pass

    # --------------------------------------------------------
    # PPO-specific aggregate metrics
    # --------------------------------------------------------
    ppo_aliases = {
        "average_weekly_turnover": "ppo_average_weekly_turnover",
        "total_transaction_costs": "ppo_total_transaction_costs",
    }

    alias = ppo_aliases.get(key)

    if alias and alias in metrics:
        try:
            return float(metrics[alias])
        except (TypeError, ValueError):
            pass

    return default

def fmt_pct(value):
    if pd.isna(value):
        return "—"
    return f"{value * 100:.2f}%"

def fmt_num(value):
    if pd.isna(value):
        return "—"
    return f"{value:.3f}"

def winner_higher(fuzzy, ppo):
    if fuzzy > ppo:
        return "Fuzzy-GA"
    if ppo > fuzzy:
        return "PPO"
    return "Tie"

def winner_lower(fuzzy, ppo):
    if fuzzy < ppo:
        return "Fuzzy-GA"
    if ppo < fuzzy:
        return "PPO"
    return "Tie"

def metric_card(label, fuzzy, ppo, formatter, winner=None):
    f = formatter(fuzzy)
    p = formatter(ppo)

    if winner == "higher":
        wf = fuzzy > ppo
        wp = ppo > fuzzy
    elif winner == "lower":
        wf = fuzzy < ppo
        wp = ppo < fuzzy
    else:
        wf = wp = False

    f_text = f'<span class="winner">{f}</span>' if wf else f
    p_text = f'<span class="winner">{p}</span>' if wp else p

    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div><b>Fuzzy-GA:</b> {f_text}</div>
            <div><b>PPO:</b> {p_text}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

def find_price_data():
    """
    Load the project's processed adjusted-price data.

    The project stores prices in wide format:
        Date | RELIANCE | HDFCBANK | TCS | ...

    Returns a long-format DataFrame so the dashboard can
    plot the six assets consistently.
    """

    price_path = DATA_DIR / "processed" / "adjusted_prices.csv"

    if not price_path.exists():
        return None

    try:
        prices = pd.read_csv(price_path)
    except Exception:
        return None

    required_columns = ["Date"] + STOCKS

    if not all(
        column in prices.columns
        for column in required_columns
    ):
        return None

    prices["Date"] = pd.to_datetime(
        prices["Date"],
        errors="coerce",
    )

    prices = prices.dropna(
        subset=["Date"]
    )

    long_prices = prices.melt(
        id_vars="Date",
        value_vars=STOCKS,
        var_name="Stock",
        value_name="Price",
    )

    long_prices["Price"] = pd.to_numeric(
        long_prices["Price"],
        errors="coerce",
    )

    long_prices = long_prices.dropna(
        subset=["Price"]
    )

    long_prices["Stock"] = (
        long_prices["Stock"]
        .astype(str)
        .str.upper()
    )

    return long_prices

def build_price_timeline(price_df, price_mode="Normalized"):
    df = price_df[
        price_df["Stock"].isin(STOCKS)
        & (price_df["Date"] >= pd.Timestamp("2025-01-01"))
        & (price_df["Date"] <= pd.Timestamp("2025-12-31"))
    ].copy()

    if df.empty:
        return None

    fig = go.Figure()

    for stock in STOCKS:
        stock_df = df[df["Stock"] == stock].sort_values("Date")

        if stock_df.empty:
            continue

        if price_mode == "Normalized":
            base = stock_df["Price"].iloc[0]
            y_values = stock_df["Price"] / base * 100
        else:
            y_values = stock_df["Price"]

        fig.add_trace(
            go.Scatter(
                x=stock_df["Date"],
                y=y_values,
                mode="lines",
                name=stock,
            )
        )

    if price_mode == "Normalized":
        title = "Relative Asset Price Performance — 2025 Test Period"
        y_title = "Indexed Price (Start = 100)"
    else:
        title = "Adjusted Stock Prices — 2025 Test Period"
        y_title = "Adjusted Price"

    fig.update_layout(
        title=title,
        xaxis_title="Date",
        yaxis_title=y_title,
        hovermode="x unified",
        height=480,
        margin=dict(l=20, r=20, t=60, b=20),
        legend=dict(
            orientation="h",
            y=1.02,
            x=0,
        ),
    )

    return fig

def calculate_drawdown(values):
    values = pd.Series(values, dtype=float)
    running_max = values.cummax()
    return values / running_max - 1.0


def rolling_volatility(returns, window=8):
    return (
        pd.Series(returns, dtype=float)
        .rolling(window)
        .std()
        * np.sqrt(52)
    )

try:
    fuzzy_raw, ppo_raw, fuzzy_metrics, ppo_metrics = load_results()
except Exception as exc:
    st.error("Dashboard data could not be loaded.")
    st.code(str(exc))
    st.stop()

fuzzy, ppo = normalize_model_data(fuzzy_raw, ppo_raw)

st.markdown(
    """
    <div class="hero">
        <h1>RL vs Fuzzy-GA for Stock Portfolio Optimization</h1>
        <p>
            Comparative evaluation of a Reinforcement Learning (PPO)
            strategy and a Fuzzy Logic–Genetic Algorithm strategy
            on the 2025 out-of-sample test period.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)

st.header("1. Data & Market Timeline")

st.markdown(
    '<div class="section-note">'
    "2025 is the common out-of-sample test period used by both strategies."
    "</div>",
    unsafe_allow_html=True,
)

price_df = find_price_data()

if price_df is not None:

    price_mode = st.radio(
        "Price display",
        ["Normalized", "Actual Adjusted Price"],
        horizontal=True,
        help=(
            "Normalized prices index every stock to 100 at the "
            "start of the test period. Actual Adjusted Price "
            "shows the original adjusted market prices."
        ),
    )

    selected_mode = (
        "Normalized"
        if price_mode == "Normalized"
        else "Actual"
    )

    price_fig = build_price_timeline(
        price_df,
        selected_mode,
    )

    if price_fig is not None:
        st.plotly_chart(
            price_fig,
            use_container_width=True,
        )

        if selected_mode == "Normalized":
            st.caption(
                "Normalized view: each asset is indexed to 100 "
                "at the beginning of the 2025 test period. "
                "This transformation is for visualization only."
            )
        else:
            st.caption(
                "Actual view: adjusted market prices from the "
                "project's processed price dataset."
            )
    else:
        st.info(
            "Market-price data was found, but no 2025 observations "
            "for the six portfolio assets were available."
        )

else:
    st.info(
        "The project's adjusted-price data could not be loaded."
    )

st.header("2. Portfolio Performance")

performance = pd.DataFrame({
    "Date": fuzzy["date"],
    "Fuzzy-GA": fuzzy["portfolio_value"],
    "PPO": ppo["portfolio_value"],
})

fig = go.Figure()

fig.add_trace(
    go.Scatter(
        x=performance["Date"],
        y=performance["Fuzzy-GA"],
        mode="lines",
        name="Fuzzy-GA",
        line=dict(width=3),
        hovertemplate="%{x|%d %b %Y}<br>Fuzzy-GA: %{y:.4f}<extra></extra>",
    )
)

fig.add_trace(
    go.Scatter(
        x=performance["Date"],
        y=performance["PPO"],
        mode="lines",
        name="PPO",
        line=dict(width=3),
        hovertemplate="%{x|%d %b %Y}<br>PPO: %{y:.4f}<extra></extra>",
    )
)

fig.add_hline(
    y=1.0,
    line_dash="dash",
    annotation_text="Initial value",
)

fig.update_layout(
    title="Cumulative Portfolio Value — Initial Value = 1.00",
    xaxis_title="Test Period",
    yaxis_title="Portfolio Value",
    hovermode="x unified",
    height=500,
    margin=dict(l=20, r=20, t=60, b=20),
    legend=dict(orientation="h", y=1.02, x=0),
)

st.plotly_chart(fig, use_container_width=True)

c1, c2, c3 = st.columns(3)
with c1:
    metric_card(
        "Total Return",
        metric_value(fuzzy_metrics, "total_return"),
        metric_value(ppo_metrics, "total_return"),
        fmt_pct,
        "higher",
    )
with c2:
    metric_card(
        "Annualized Return",
        metric_value(fuzzy_metrics, "annualized_return"),
        metric_value(ppo_metrics, "annualized_return"),
        fmt_pct,
        "higher",
    )
with c3:
    metric_card(
        "Sharpe Ratio",
        metric_value(fuzzy_metrics, "sharpe_ratio"),
        metric_value(ppo_metrics, "sharpe_ratio"),
        fmt_num,
        "higher",
    )

c4, c5, c6 = st.columns(3)
with c4:
    metric_card(
        "Annualized Volatility",
        metric_value(fuzzy_metrics, "annualized_volatility"),
        metric_value(ppo_metrics, "annualized_volatility"),
        fmt_pct,
        "lower",
    )
with c5:
    metric_card(
        "Maximum Drawdown",
        metric_value(fuzzy_metrics, "maximum_drawdown"),
        metric_value(ppo_metrics, "maximum_drawdown"),
        fmt_pct,
        "higher",
    )
with c6:
    metric_card(
        "Average Weekly Turnover",
        metric_value(fuzzy_metrics, "average_weekly_turnover"),
        metric_value(ppo_metrics, "average_weekly_turnover"),
        fmt_pct,
        "lower",
    )

st.header("3. Risk Analysis")

risk_df = pd.DataFrame({
    "Date": fuzzy["date"],
    "Fuzzy-GA Value": fuzzy["portfolio_value"],
    "PPO Value": ppo["portfolio_value"],
    "Fuzzy-GA Return": fuzzy["return"],
    "PPO Return": ppo["return"],
})

risk_df["Fuzzy-GA Drawdown"] = calculate_drawdown(
    risk_df["Fuzzy-GA Value"]
)
risk_df["PPO Drawdown"] = calculate_drawdown(
    risk_df["PPO Value"]
)

risk_df["Fuzzy-GA Rolling Volatility"] = rolling_volatility(
    risk_df["Fuzzy-GA Return"]
)
risk_df["PPO Rolling Volatility"] = rolling_volatility(
    risk_df["PPO Return"]
)

drawdown_fig = go.Figure()

drawdown_fig.add_trace(
    go.Scatter(
        x=risk_df["Date"],
        y=risk_df["Fuzzy-GA Drawdown"] * 100,
        mode="lines",
        name="Fuzzy-GA",
        line=dict(width=2.5),
    )
)

drawdown_fig.add_trace(
    go.Scatter(
        x=risk_df["Date"],
        y=risk_df["PPO Drawdown"] * 100,
        mode="lines",
        name="PPO",
        line=dict(width=2.5),
    )
)

drawdown_fig.update_layout(
    title="Portfolio Drawdown",
    xaxis_title="Date",
    yaxis_title="Drawdown (%)",
    hovermode="x unified",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(drawdown_fig, use_container_width=True)

vol_fig = go.Figure()

vol_fig.add_trace(
    go.Scatter(
        x=risk_df["Date"],
        y=risk_df["Fuzzy-GA Rolling Volatility"] * 100,
        mode="lines",
        name="Fuzzy-GA",
    )
)

vol_fig.add_trace(
    go.Scatter(
        x=risk_df["Date"],
        y=risk_df["PPO Rolling Volatility"] * 100,
        mode="lines",
        name="PPO",
    )
)

vol_fig.update_layout(
    title="Rolling Annualized Volatility (8 Weeks)",
    xaxis_title="Date",
    yaxis_title="Annualized Volatility (%)",
    hovermode="x unified",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(vol_fig, use_container_width=True)

dist_fig = go.Figure()

dist_fig.add_trace(
    go.Histogram(
        x=risk_df["Fuzzy-GA Return"] * 100,
        name="Fuzzy-GA",
        opacity=0.65,
        nbinsx=18,
    )
)

dist_fig.add_trace(
    go.Histogram(
        x=risk_df["PPO Return"] * 100,
        name="PPO",
        opacity=0.65,
        nbinsx=18,
    )
)

dist_fig.update_layout(
    title="Weekly Return Distribution",
    xaxis_title="Weekly Return (%)",
    yaxis_title="Frequency",
    barmode="overlay",
    height=420,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(dist_fig, use_container_width=True)

st.header("4. Portfolio Allocation")

allocation_model = st.radio(
    "Select strategy",
    ["Fuzzy-GA", "PPO"],
    horizontal=True,
)

allocation_dates = (
    fuzzy_raw["holding_end"]
    if allocation_model == "Fuzzy-GA"
    else ppo_raw["Holding_End"]
)

allocation_dates = pd.to_datetime(
    allocation_dates
).sort_values().unique()

selected_date = st.select_slider(
    "Select test-period date",
    options=list(allocation_dates),
    format_func=lambda x: pd.Timestamp(x).strftime(
        "%d %b %Y"
    ),
)

if allocation_model == "Fuzzy-GA":
    row = fuzzy_raw.loc[
        fuzzy_raw["holding_end"] == pd.Timestamp(selected_date)
    ].iloc[0]

    target_cols = [
        f"target_weight_{stock}"
        for stock in STOCKS
    ]

    ending_cols = [
        f"ending_weight_{stock}"
        for stock in STOCKS
    ]

else:
    row = ppo_raw.loc[
        ppo_raw["Holding_End"] == pd.Timestamp(selected_date)
    ].iloc[0]

    target_cols = [
        f"Weight_{stock}"
        for stock in STOCKS
    ]

    ending_cols = [
        f"PPO_EndingWeight_{stock}"
        for stock in STOCKS
    ]

allocation = pd.DataFrame({
    "Stock": STOCKS,
    "Target Weight": [
        float(row[col]) for col in target_cols
    ],
    "Ending Weight": [
        float(row[col]) for col in ending_cols
    ],
})

allocation_fig = go.Figure()

allocation_fig.add_trace(
    go.Bar(
        x=allocation["Stock"],
        y=allocation["Target Weight"] * 100,
        name="Target Weight",
    )
)

allocation_fig.add_trace(
    go.Bar(
        x=allocation["Stock"],
        y=allocation["Ending Weight"] * 100,
        name="Ending Weight",
    )
)

allocation_fig.update_layout(
    title=(
        f"{allocation_model} Allocation — "
        f"{pd.Timestamp(selected_date).strftime('%d %b %Y')}"
    ),
    xaxis_title="Asset",
    yaxis_title="Portfolio Weight (%)",
    barmode="group",
    height=430,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(
    allocation_fig,
    use_container_width=True,
)

st.subheader("Portfolio Weight Evolution")

weight_fig = go.Figure()

for stock in STOCKS:
    if allocation_model == "Fuzzy-GA":
        values = fuzzy_raw[f"target_weight_{stock}"] * 100
        dates = fuzzy_raw["holding_end"]
    else:
        values = ppo_raw[f"Weight_{stock}"] * 100
        dates = ppo_raw["Holding_End"]

    weight_fig.add_trace(
        go.Scatter(
            x=dates,
            y=values,
            mode="lines",
            stackgroup="one",
            name=stock,
        )
    )

weight_fig.update_layout(
    title=f"{allocation_model} Target Portfolio Weights Over Time",
    xaxis_title="Date",
    yaxis_title="Portfolio Weight (%)",
    hovermode="x unified",
    height=500,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(
    weight_fig,
    use_container_width=True,
)

st.header("5. Trading / Rebalancing Behaviour")

turnover_df = pd.DataFrame({
    "Date": fuzzy["date"],
    "Fuzzy-GA": fuzzy["turnover"] * 100,
    "PPO": ppo["turnover"] * 100,
})

turnover_fig = go.Figure()

turnover_fig.add_trace(
    go.Scatter(
        x=turnover_df["Date"],
        y=turnover_df["Fuzzy-GA"],
        mode="lines+markers",
        name="Fuzzy-GA",
    )
)

turnover_fig.add_trace(
    go.Scatter(
        x=turnover_df["Date"],
        y=turnover_df["PPO"],
        mode="lines+markers",
        name="PPO",
    )
)

turnover_fig.update_layout(
    title="Weekly Portfolio Turnover",
    xaxis_title="Date",
    yaxis_title="One-Way Turnover (%)",
    hovermode="x unified",
    height=450,
    margin=dict(l=20, r=20, t=60, b=20),
)

st.plotly_chart(
    turnover_fig,
    use_container_width=True,
)

t1, t2, t3, t4 = st.columns(4)

fuzzy_avg_turnover = metric_value(
    fuzzy_metrics,
    "average_weekly_turnover",
)
ppo_avg_turnover = metric_value(
    ppo_metrics,
    "average_weekly_turnover",
)

fuzzy_cost = metric_value(
    fuzzy_metrics,
    "total_transaction_costs",
)
ppo_cost = metric_value(
    ppo_metrics,
    "total_transaction_costs",
)

with t1:
    st.metric(
        "Fuzzy-GA Avg. Weekly Turnover",
        fmt_pct(fuzzy_avg_turnover),
    )

with t2:
    st.metric(
        "PPO Avg. Weekly Turnover",
        fmt_pct(ppo_avg_turnover),
    )

with t3:
    st.metric(
        "Fuzzy-GA Transaction Costs",
        fmt_pct(fuzzy_cost),
    )

with t4:
    st.metric(
        "PPO Transaction Costs",
        fmt_pct(ppo_cost),
    )

st.header("6. Final Results / Verdict")

comparison = pd.DataFrame({
    "Metric": [
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Sharpe Ratio",
        "Maximum Drawdown",
        "Average Weekly Turnover",
        "Transaction Costs",
    ],
    "Fuzzy-GA": [
        metric_value(fuzzy_metrics, "total_return"),
        metric_value(fuzzy_metrics, "annualized_return"),
        metric_value(fuzzy_metrics, "annualized_volatility"),
        metric_value(fuzzy_metrics, "sharpe_ratio"),
        metric_value(fuzzy_metrics, "maximum_drawdown"),
        metric_value(fuzzy_metrics, "average_weekly_turnover"),
        metric_value(fuzzy_metrics, "total_transaction_costs"),
    ],
    "PPO": [
        metric_value(ppo_metrics, "total_return"),
        metric_value(ppo_metrics, "annualized_return"),
        metric_value(ppo_metrics, "annualized_volatility"),
        metric_value(ppo_metrics, "sharpe_ratio"),
        metric_value(ppo_metrics, "average_weekly_turnover"),
        metric_value(ppo_metrics, "average_weekly_turnover"),
        metric_value(ppo_metrics, "total_transaction_costs"),
    ],
})

# Correct the PPO maximum drawdown row explicitly.
comparison.loc[
    comparison["Metric"] == "Maximum Drawdown",
    "PPO",
] = metric_value(
    ppo_metrics,
    "maximum_drawdown",
)

comparison["Winner"] = [
    winner_higher(
        comparison.loc[0, "Fuzzy-GA"],
        comparison.loc[0, "PPO"],
    ),
    winner_higher(
        comparison.loc[1, "Fuzzy-GA"],
        comparison.loc[1, "PPO"],
    ),
    winner_lower(
        comparison.loc[2, "Fuzzy-GA"],
        comparison.loc[2, "PPO"],
    ),
    winner_higher(
        comparison.loc[3, "Fuzzy-GA"],
        comparison.loc[3, "PPO"],
    ),
    winner_higher(
        comparison.loc[4, "Fuzzy-GA"],
        comparison.loc[4, "PPO"],
    ),
    winner_lower(
        comparison.loc[5, "Fuzzy-GA"],
        comparison.loc[5, "PPO"],
    ),
    winner_lower(
        comparison.loc[6, "Fuzzy-GA"],
        comparison.loc[6, "PPO"],
    ),
]

display_rows = []

for i in range(len(comparison)):
    metric = comparison.loc[i, "Metric"]
    fuzzy_value = comparison.loc[i, "Fuzzy-GA"]
    ppo_value = comparison.loc[i, "PPO"]
    winner = comparison.loc[i, "Winner"]

    if metric in {
        "Total Return",
        "Annualized Return",
        "Annualized Volatility",
        "Maximum Drawdown",
        "Average Weekly Turnover",
        "Transaction Costs",
    }:
        fuzzy_display = fmt_pct(fuzzy_value)
        ppo_display = fmt_pct(ppo_value)
    else:
        fuzzy_display = fmt_num(fuzzy_value)
        ppo_display = fmt_num(ppo_value)

    display_rows.append({
        "Metric": metric,
        "Fuzzy-GA": fuzzy_display,
        "PPO": ppo_display,
        "Winner": winner,
    })

display_table = pd.DataFrame(display_rows)

st.dataframe(
    display_table,
    use_container_width=True,
    hide_index=True,
)

f_total = metric_value(fuzzy_metrics, "total_return")
p_total = metric_value(ppo_metrics, "total_return")

f_sharpe = metric_value(fuzzy_metrics, "sharpe_ratio")
p_sharpe = metric_value(ppo_metrics, "sharpe_ratio")

f_vol = metric_value(
    fuzzy_metrics,
    "annualized_volatility",
)
p_vol = metric_value(
    ppo_metrics,
    "annualized_volatility",
)

f_dd = metric_value(
    fuzzy_metrics,
    "maximum_drawdown",
)
p_dd = metric_value(
    ppo_metrics,
    "maximum_drawdown",
)

f_turnover = metric_value(
    fuzzy_metrics,
    "average_weekly_turnover",
)
p_turnover = metric_value(
    ppo_metrics,
    "average_weekly_turnover",
)

st.subheader("Overall Interpretation")

if p_sharpe > f_sharpe:
    risk_adjusted_winner = "PPO"
else:
    risk_adjusted_winner = "Fuzzy-GA"

if p_total > f_total:
    return_winner = "PPO"
else:
    return_winner = "Fuzzy-GA"

if f_vol < p_vol:
    volatility_winner = "Fuzzy-GA"
else:
    volatility_winner = "PPO"

if f_turnover < p_turnover:
    turnover_winner = "Fuzzy-GA"
else:
    turnover_winner = "PPO"

if p_dd > f_dd:
    drawdown_winner = "PPO"
else:
    drawdown_winner = "Fuzzy-GA"

st.info(
    f"""
**PPO achieved the stronger overall test-period investment performance.**

- **Return:** {return_winner} achieved the higher total return ({fmt_pct(max(f_total, p_total))}).
- **Risk-adjusted performance:** {risk_adjusted_winner} achieved the higher Sharpe ratio.
- **Volatility:** {volatility_winner} had the lower annualized volatility.
- **Maximum drawdown:** {drawdown_winner} had the better maximum drawdown.
- **Trading efficiency:** {turnover_winner} had the lower average weekly turnover.
"""
)