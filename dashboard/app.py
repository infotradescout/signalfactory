"""
dashboard/app.py — Streamlit interactive dashboard.

Run with:
    streamlit run dashboard/app.py
"""
import sys
from pathlib import Path

# Allow importing src/ from within the dashboard directory
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import logging
import warnings
import json
import shutil
import re
from datetime import date, datetime

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
import yaml

from src.data import DataCache, FREDConnector, FileLoader, WorldBankConnector
from src.data.scraper import WebScraper
from src.features import FeaturePipeline
from src.models import ModelRegistry
from src.outputs import ReportBuilder

warnings.filterwarnings("ignore")
logging.basicConfig(level=logging.WARNING)

SCAFFOLD_TARGET_KEYS = [
    "food_demand_weekly",
    "construction_material_cost_direction",
    "restaurant_failure_risk",
    "construction_project_delay_risk",
]

SCAFFOLD_TEMPLATE_FILES = {
    "food_demand_weekly": "food_demand_weekly_template.csv",
    "construction_material_cost_direction": "construction_material_cost_direction_template.csv",
    "restaurant_failure_risk": "restaurant_failure_risk_template.csv",
    "construction_project_delay_risk": "construction_project_delay_risk_template.csv",
}

# ─── page config ─────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SignalFactory",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── helpers ─────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
def load_config():
    targets_path = PROJECT_ROOT / "config" / "targets.yaml"
    sources_path = PROJECT_ROOT / "config" / "sources.yaml"
    with open(targets_path) as f:
        targets = yaml.safe_load(f)
    with open(sources_path) as f:
        sources = yaml.safe_load(f)
    return targets, sources


def build_cache() -> DataCache:
    return DataCache(str(PROJECT_ROOT / "data" / "cache" / "store.db"))


def _uploads_dir() -> Path:
    path = PROJECT_ROOT / "data" / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _templates_dir() -> Path:
    return PROJECT_ROOT / "data" / "templates"


def _history_path() -> Path:
    path = PROJECT_ROOT / "data" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path / "run_history.csv"


def _outcomes_path() -> Path:
    path = PROJECT_ROOT / "data" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path / "outcome_log.csv"


def _execution_tracker_path() -> Path:
    path = PROJECT_ROOT / "data" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path / "execution_tracker.csv"


def _load_demo_templates() -> tuple[int, int]:
    """Copy scaffold template files into uploads directory (skip existing files)."""
    src_dir = _templates_dir()
    dst_dir = _uploads_dir()
    if not src_dir.exists():
        return 0, 0

    copied = 0
    skipped = 0
    for filename in SCAFFOLD_TEMPLATE_FILES.values():
        src = src_dir / filename
        dst = dst_dir / filename
        if not src.exists():
            continue
        if dst.exists():
            skipped += 1
            continue
        shutil.copy2(src, dst)
        copied += 1
    return copied, skipped


def _uploaded_data_count() -> int:
    upload_dir = _uploads_dir()
    supported = {".csv", ".tsv", ".xlsx", ".xls", ".parquet"}
    return len([p for p in upload_dir.iterdir() if p.is_file() and p.suffix.lower() in supported])


def _append_run_history(summary_df: pd.DataFrame) -> None:
    if summary_df.empty:
        return
    history_file = _history_path()
    to_save = summary_df.copy()
    to_save.insert(0, "run_at", datetime.utcnow().isoformat())
    if history_file.exists():
        to_save.to_csv(history_file, mode="a", header=False, index=False)
    else:
        to_save.to_csv(history_file, index=False)


def _load_recent_history(max_rows: int = 12) -> pd.DataFrame:
    history_file = _history_path()
    if not history_file.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(history_file)
        return df.tail(max_rows)
    except Exception:
        return pd.DataFrame()


def _append_outcome_log(row: dict) -> None:
    outcomes_file = _outcomes_path()
    frame = pd.DataFrame([row])
    if outcomes_file.exists():
        frame.to_csv(outcomes_file, mode="a", header=False, index=False)
    else:
        frame.to_csv(outcomes_file, index=False)


def _load_outcomes(max_rows: int = 200) -> pd.DataFrame:
    outcomes_file = _outcomes_path()
    if not outcomes_file.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(outcomes_file)
        return df.tail(max_rows)
    except Exception:
        return pd.DataFrame()


def _load_execution_tracker() -> pd.DataFrame:
    tracker_file = _execution_tracker_path()
    if not tracker_file.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(tracker_file)
    except Exception:
        return pd.DataFrame()


def _save_execution_tracker(df: pd.DataFrame) -> None:
    tracker_file = _execution_tracker_path()
    df.to_csv(tracker_file, index=False)


def _seed_execution_tracker(execution_df: pd.DataFrame) -> pd.DataFrame:
    if execution_df.empty:
        return _load_execution_tracker()

    seeded = execution_df.copy()
    if "target" in seeded.columns:
        seeded["target_key"] = seeded["target"].astype(str).str.lower().str.replace(" ", "_", regex=False)
    seeded["owner"] = seeded.get("owner", "Unassigned").fillna("Unassigned")
    seeded["due_date"] = seeded.get("due_date", "")
    seeded["status"] = seeded.get("status", "todo").fillna("todo")

    existing = _load_execution_tracker()
    if existing.empty:
        return seeded

    merged = existing.set_index("target_key").combine_first(seeded.set_index("target_key")).reset_index()
    for col in seeded.columns:
        if col in merged.columns:
            merged[col] = merged[col].fillna(seeded.set_index("target_key").reindex(merged["target_key"])[col].values)
    return merged


def _validation_snapshot(outcomes_df: pd.DataFrame) -> dict:
    if outcomes_df.empty or "logged_at" not in outcomes_df.columns:
        return {
            "rows_4w": 0,
            "hit_rate_4w": 0.0,
            "paper_only": True,
            "note": "No tracked outcomes yet.",
        }

    df = outcomes_df.copy()
    df["logged_at"] = pd.to_datetime(df["logged_at"], errors="coerce")
    cutoff = pd.Timestamp.utcnow() - pd.Timedelta(days=28)
    df = df[df["logged_at"] >= cutoff]

    if df.empty:
        return {
            "rows_4w": 0,
            "hit_rate_4w": 0.0,
            "paper_only": True,
            "note": "No outcomes in last 4 weeks.",
        }

    hit_map = {"hit": 1.0, "partial": 0.5, "miss": 0.0}
    score = df["outcome_grade"].map(hit_map).fillna(0.0)
    hit_rate = float(score.mean())
    rows = int(len(df))

    paper_only = rows < 20 or hit_rate < 0.60
    note = (
        "Paper mode only (need >=20 outcomes and >=60% hit rate)."
        if paper_only
        else "Eligible for small live tests with strict risk limits."
    )
    return {
        "rows_4w": rows,
        "hit_rate_4w": hit_rate,
        "paper_only": paper_only,
        "note": note,
    }


def run_prediction(target_cfg: dict, sources_cfg: dict, countries: list[str],
                   start_year: int, end_year: int, horizon: int, show_errors: bool = True):
    cache = build_cache()
    feature_groups = target_cfg.get("features", {})

    # ── World Bank ────────────────────────────────────────────────────────
    wb_df = pd.DataFrame()
    if sources_cfg.get("world_bank", {}).get("enabled", False):
        wb_indicators = sources_cfg["world_bank"].get("indicators", {})
        needed_wb = {
            name: code
            for name, code in wb_indicators.items()
            if any(name in g for g in feature_groups.values())
        }
        if needed_wb:
            wb = WorldBankConnector(
                indicators=needed_wb,
                cache=cache,
                cache_days=sources_cfg["world_bank"].get("cache_days", 7),
            )
            with st.spinner("Fetching World Bank data…"):
                wb_df = wb.fetch(countries, start_year, end_year)

    # ── FRED ──────────────────────────────────────────────────────────────
    fred_df = pd.DataFrame()
    if sources_cfg.get("fred", {}).get("enabled", False):
        fred_series = sources_cfg["fred"].get("series", {})
        needed_fred = {
            name: sid
            for name, sid in fred_series.items()
            if any(name in g for g in feature_groups.values())
        }
        if needed_fred:
            fred = FREDConnector(
                series=needed_fred,
                cache=cache,
                cache_days=sources_cfg["fred"].get("cache_days", 1),
            )
            with st.spinner("Fetching FRED data…"):
                fred_df = fred.fetch(
                    f"{start_year}-01-01",
                    f"{end_year}-12-31",
                    frequency="a",
                )

    # ── User uploads (CSV / Excel / Parquet) ───────────────────────────
    extra_dfs = []
    if sources_cfg.get("csv", {}).get("enabled", False):
        loader = FileLoader(
            upload_dir=str(PROJECT_ROOT / "data" / "uploads"),
            cache=cache,
        )
        extra_dfs = list(loader.load_all().values())

    # ── feature pipeline ──────────────────────────────────────────────────
    pipe = FeaturePipeline(target_cfg)
    X, y = pipe.fit_transform(wb_df=wb_df, fred_df=fred_df, extra_dfs=extra_dfs)

    if X.empty:
        if show_errors:
            st.error("No data was retrieved. Check API keys and network connectivity.")
        return None, None, None

    # ── model ─────────────────────────────────────────────────────────────
    registry = ModelRegistry()
    result = registry.run(target_cfg, X, y, horizon=horizon)
    return result, X, wb_df if not wb_df.empty else fred_df


def _extract_horizon_from_question(question: str) -> int:
    q = question.lower()
    m = re.search(r"(\d+)\s*(week|weeks|month|months|year|years)", q)
    if not m:
        return 12
    value = int(m.group(1))
    unit = m.group(2)
    if "week" in unit:
        return max(1, min(52, value))
    if "month" in unit:
        return max(1, min(36, value))
    return max(1, min(36, value * 12))


def _route_question_to_target(question: str, targets_cfg: dict) -> dict:
    q = question.lower().strip()
    routes = [
        ("food_demand_weekly", ["food", "demand", "sales", "restaurant", "inventory"]),
        ("construction_material_cost_direction", ["material", "lumber", "steel", "cement", "diesel", "cost"]),
        ("restaurant_failure_risk", ["restaurant", "failure", "close", "shutdown", "risk"]),
        ("construction_project_delay_risk", ["project", "delay", "construction", "schedule", "late"]),
        ("inflation_forecast", ["inflation", "cpi", "price level"]),
        ("market_trend", ["stock", "market", "s&p", "nasdaq", "equity"]),
    ]

    best = None
    best_score = 0
    for key, words in routes:
        if key not in targets_cfg.get("targets", {}):
            continue
        score = sum(1 for w in words if w in q)
        if score > best_score:
            best_score = score
            best = key

    if best and best_score > 0:
        return {
            "status": "supported",
            "target_key": best,
            "score": best_score,
        }
    return {
        "status": "unsupported",
        "target_key": None,
        "score": 0,
    }


def _question_needs_plan(question: str) -> dict:
    q = question.lower()
    domain_sources = ["CSV exports", "Public APIs", "Manual labels"]
    example_features = ["time", "location/entity", "outcome label", "3-10 predictor fields"]
    target_type = "classification"

    if any(k in q for k in ["price", "rate", "forecast", "how much", "level", "value"]):
        target_type = "forecast"
    if any(k in q for k in ["win", "lose", "yes", "no", "will", "happen"]):
        target_type = "classification"

    if any(k in q for k in ["weather", "rain", "storm", "temperature"]):
        domain_sources = ["NOAA/NWS APIs", "Historical weather datasets", "Location metadata"]
        example_features = ["date", "location", "temperature", "precipitation", "wind", "historical outcome"]
    elif any(k in q for k in ["election", "poll", "politics", "policy"]):
        domain_sources = ["Polling APIs", "Economic indicators", "Election archives"]
        example_features = ["date", "region", "poll spread", "approval", "economic trend", "resolved result"]
    elif any(k in q for k in ["sport", "team", "game", "match"]):
        domain_sources = ["Sports stats APIs", "Odds history", "Injury/news feeds"]
        example_features = ["date", "teams", "recent form", "injuries", "odds", "final result"]
    elif any(k in q for k in ["crypto", "bitcoin", "eth", "token"]):
        domain_sources = ["Exchange OHLCV APIs", "On-chain metrics", "Macro indicators"]
        example_features = ["timestamp", "price", "volume", "volatility", "funding rates", "target outcome"]
    elif any(k in q for k in ["market", "stock", "bond", "fed", "kalshi"]):
        domain_sources = ["FRED", "Exchange data APIs", "Event resolution source"]
        example_features = ["timestamp", "market price", "macro factors", "event rule", "resolved outcome"]

    return {
        "target_type": target_type,
        "horizon": _extract_horizon_from_question(question),
        "minimum_history": "At least 2 years or 50+ resolved events",
        "sources": domain_sources,
        "features": example_features,
    }


def _build_target_snippet_from_question(question: str, plan: dict) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", question.lower()).strip("_")
    slug = slug[:48] if slug else "new_question_target"
    return (
        f"{slug}:\n"
        f"  label: \"{question[:80]}\"\n"
        f"  type: {plan['target_type']}\n"
        f"  output_unit: \"custom\"\n"
        f"  target_column: outcome_label\n"
        f"  features:\n"
        f"    economic: [feature_1, feature_2]\n"
        f"    social: [feature_3]\n"
        f"    market: [feature_4]\n"
        f"  countries: [US]\n"
        f"  date_range:\n"
        f"    start: \"2018-01-01\"\n"
        f"    end: \"today\"\n"
        f"  model: {'arima' if plan['target_type'] == 'forecast' else 'random_forest'}\n"
        f"  forecast_horizon: {plan['horizon']}\n"
    )


# ─── sidebar ─────────────────────────────────────────────────────────────────

def sidebar(targets_cfg: dict, sources_cfg: dict):
    st.sidebar.header("Configuration")
    simple_mode = st.sidebar.checkbox("Simple Mode (recommended)", value=True)
    decision_policy = st.sidebar.selectbox(
        "Decision Policy",
        ["Conservative", "Balanced", "Aggressive"],
        index=1,
        help="Controls urgency and confidence thresholds for recommendations.",
    )

    if st.sidebar.button("Load Demo Data (one click)", use_container_width=True):
        copied, skipped = _load_demo_templates()
        st.sidebar.success(f"Demo data ready. Added {copied} files, kept {skipped} existing.")

    st.sidebar.divider()

    target_names = list(targets_cfg["targets"].keys())
    default_idx = target_names.index(targets_cfg.get("active_target", target_names[0]))
    target_key = st.sidebar.selectbox("Prediction Target", target_names, index=default_idx)
    target_cfg = targets_cfg["targets"][target_key]
    target_cfg["id"] = target_key

    st.sidebar.markdown(f"**{target_cfg['label']}**  \n`{target_cfg['type']}`")
    st.sidebar.divider()

    all_countries = target_cfg.get("countries", ["US"])
    default_start = int(target_cfg.get("date_range", {}).get("start", "2000")[:4])
    end_year = date.today().year

    if simple_mode:
        countries = all_countries[:1] if all_countries else ["US"]
        start_year = default_start
        horizon = int(target_cfg.get("forecast_horizon", 12)) if target_cfg["type"] == "forecast" else 1
        st.sidebar.caption("Simple mode uses safe defaults automatically.")
    else:
        countries = st.sidebar.multiselect("Countries", all_countries, default=all_countries[:3])
        if not countries:
            countries = all_countries[:1]
        start_year = st.sidebar.slider("Start year", 1990, date.today().year - 2, default_start)
        horizon = 1
        if target_cfg["type"] == "forecast":
            horizon = st.sidebar.slider(
                "Forecast horizon (periods)",
                1,
                36,
                int(target_cfg.get("forecast_horizon", 12)),
            )

    run = st.sidebar.button("▶ Run Prediction", type="primary", use_container_width=True)
    run_all = st.sidebar.button("▶ Run All 4 Scaffold Targets", use_container_width=True)
    return (
        target_cfg,
        countries,
        start_year,
        end_year,
        horizon,
        run,
        run_all,
        simple_mode,
        decision_policy,
    )


# ─── main UI ─────────────────────────────────────────────────────────────────

def main():
    _init_state()

    st.title("📊 SignalFactory")
    st.caption(
        "Analyse economics, culture, and market factors to predict virtually anything."
    )

    st.subheader("Autopilot (Hands-Off)")
    col_a, col_b, col_c, col_d = st.columns(4)
    with col_a:
        if st.button("1) Prepare Demo Data", use_container_width=True):
            copied, skipped = _load_demo_templates()
            st.success(f"Demo data ready. Added {copied} files, kept {skipped} existing.")
    with col_b:
        run_all_autopilot = st.button("2) Run All 4 Now", type="primary", use_container_width=True)
    with col_c:
        st.download_button(
            "3) Download Executive Summary",
            data=st.session_state["batch_executive_text"],
            file_name="executive_summary.txt",
            mime="text/plain",
            use_container_width=True,
            disabled=not bool(st.session_state["batch_executive_text"]),
        )
    with col_d:
        do_everything = st.button("Do Everything For Me", use_container_width=True)

    st.caption("Decision-grade mode: outputs include action, confidence, urgency, and timeframe.")

    targets_cfg, sources_cfg = load_config()

    st.subheader("Ask A Question")
    question = st.text_input(
        "Type the prediction question you want answered",
        placeholder="Example: Will construction material costs go up over the next 3 months?",
    )
    ask_now = st.button("Answer This Question", type="primary", use_container_width=True)

    if ask_now and question.strip():
        route = _route_question_to_target(question, targets_cfg)
        if route["status"] == "supported":
            key = route["target_key"]
            cfg = dict(targets_cfg["targets"][key])
            cfg["id"] = key
            countries_q = cfg.get("countries", ["US"])
            start_year_q = int(cfg.get("date_range", {}).get("start", "2000")[:4])
            end_year_q = date.today().year
            horizon_q = _extract_horizon_from_question(question)
            if cfg.get("type") != "forecast":
                horizon_q = 1

            result_q, X_q, _ = run_prediction(
                cfg,
                sources_cfg,
                countries_q,
                start_year_q,
                end_year_q,
                horizon_q,
                show_errors=True,
            )
            if result_q is not None:
                conf_q = _estimate_confidence(result_q, cfg.get("type", ""))
                decision_q = _decision_packet(key, result_q, _latest_decision_value(result_q), conf_q, "Balanced")
                st.success("Question answered with existing model support.")
                st.write(_single_run_narrative(cfg, result_q, conf_q, decision_q, X_q))
            else:
                st.warning("Question matched a target, but current data is insufficient to produce a result.")
        else:
            needs = _question_needs_plan(question)
            st.warning("This question is not covered yet. Here is exactly what is needed to answer it.")
            st.markdown(
                f"**Recommended Model Type:** {needs['target_type']}  \n"
                f"**Suggested Horizon:** {needs['horizon']} periods  \n"
                f"**Minimum History Needed:** {needs['minimum_history']}"
            )
            st.write("Needed data sources:")
            for src in needs["sources"]:
                st.write(f"- {src}")
            st.write("Required fields:")
            for f in needs["features"]:
                st.write(f"- {f}")

            snippet = _build_target_snippet_from_question(question, needs)
            st.write("Scaffold snippet to add when data is ready:")
            st.code(snippet, language="yaml")

    (
        target_cfg,
        countries,
        start_year,
        end_year,
        horizon,
        run,
        run_all,
        simple_mode,
        decision_policy,
    ) = sidebar(targets_cfg, sources_cfg)

    st.info("Hands-off mode is available: load demo data, then click one run button.")
    if _uploaded_data_count() == 0:
        st.warning("No uploaded datasets found yet.")
        if st.button("Prepare Demo Data Now", type="primary"):
            copied, skipped = _load_demo_templates()
            st.success(f"Demo data ready. Added {copied} files, kept {skipped} existing.")

    recent = _load_recent_history()
    if not recent.empty:
        with st.expander("Recent Runs", expanded=False):
            st.dataframe(recent, use_container_width=True)

    outcomes_df = _load_outcomes()
    validation = _validation_snapshot(outcomes_df)
    st.subheader("Deployment Readiness")
    mode_label = "Paper Mode" if validation["paper_only"] else "Limited Live Mode"
    c1, c2, c3 = st.columns(3)
    c1.metric("Mode", mode_label)
    c2.metric("4-Week Hit Rate", f"{validation['hit_rate_4w'] * 100:.0f}%")
    c3.metric("Tracked Outcomes (4w)", str(validation["rows_4w"]))
    if validation["paper_only"]:
        st.warning(validation["note"])
    else:
        st.success(validation["note"])

    if do_everything:
        if _uploaded_data_count() == 0:
            _load_demo_templates()
        _run_all_scaffold_targets(targets_cfg, sources_cfg, decision_policy)
        return

    if run_all_autopilot:
        _run_all_scaffold_targets(targets_cfg, sources_cfg, decision_policy)
        return

    if not run and not run_all:
        st.info(
            "👈 Select a prediction target in the sidebar and click **Run Prediction**.\n\n"
            "**Available targets:**\n" +
            "\n".join(
                f"- `{k}` — {v['label']} ({v['type']})"
                for k, v in targets_cfg["targets"].items()
            )
        )
        _show_upload_panel()
        return

    if run_all:
        _run_all_scaffold_targets(targets_cfg, sources_cfg, decision_policy)
        return

    result, X, raw_df = run_prediction(
        target_cfg, sources_cfg, countries, start_year, end_year, horizon
    )
    if result is None:
        return

    # ── layout ────────────────────────────────────────────────────────────
    col1, col2, col3 = st.columns(3)

    latest = result.latest_prediction()
    col1.metric(
        label=target_cfg["label"],
        value=f"{latest:.2f} {target_cfg.get('output_unit', '')}" if isinstance(latest, float)
              else str(latest),
    )
    col2.metric("Model", result.model_type.split("/")[-1])
    if result.metrics:
        primary_metric = next(iter(result.metrics))
        col3.metric(primary_metric, f"{result.metrics[primary_metric]:.4f}")

    if simple_mode:
        confidence = _estimate_confidence(result, target_cfg.get("type", ""))
        decision = _decision_packet(
            target_cfg.get("id", ""), result, latest, confidence, decision_policy
        )
        st.success(_human_summary(target_cfg.get("id", ""), latest, decision))
        st.markdown(
            f"**Recommended Action:** {decision['action']}  \n"
            f"**Urgency:** {decision['urgency']}  \n"
            f"**Timeframe:** {decision['timeframe']}  \n"
            f"**Confidence:** {confidence * 100:.0f}%"
        )
        st.subheader("Natural Language Brief")
        st.write(_single_run_narrative(target_cfg, result, confidence, decision, X))

    st.subheader("Prediction Market Decision Card")
    st.caption("Use this to turn model output into an actionable YES/NO trade plan.")
    mk_col1, mk_col2, mk_col3 = st.columns(3)
    with mk_col1:
        market_yes_cents = st.number_input(
            "Current market YES price (cents)",
            min_value=1,
            max_value=99,
            value=50,
            step=1,
        )
    with mk_col2:
        strategy_bankroll = st.number_input(
            "Strategy bankroll ($)",
            min_value=100,
            max_value=1000000,
            value=5000,
            step=100,
        )
    with mk_col3:
        max_risk_pct = st.slider("Max risk per trade (%)", min_value=1, max_value=10, value=2)

    model_prob = _estimate_market_probability(target_cfg.get("id", ""), result)
    market_prob = float(market_yes_cents) / 100.0
    card = _build_market_decision(
        model_prob=model_prob,
        market_prob=market_prob,
        bankroll=float(strategy_bankroll),
        max_risk_pct=float(max_risk_pct) / 100.0,
    )

    st.markdown(
        f"**Model Probability (YES):** {model_prob * 100:.1f}%  \n"
        f"**Market Implied Probability (YES):** {market_prob * 100:.1f}%  \n"
        f"**Edge:** {card['edge_pct']:+.1f}%  \n"
        f"**Suggested Side:** {card['side']}  \n"
        f"**Suggested Max Stake:** ${card['stake_usd']:.0f}"
    )
    st.info(card["narrative"])

    tabs = st.tabs(["Predictions", "Feature Importance", "Raw Data", "Metrics", "Export"])

    # ── Predictions tab ───────────────────────────────────────────────────
    with tabs[0]:
        if result.forecast is not None:
            _plot_forecast(result)
        elif result.probabilities is not None:
            _plot_probabilities(result)
        else:
            _plot_regression(result)

    # ── Feature Importance tab ────────────────────────────────────────────
    with tabs[1]:
        if result.feature_importance is not None and not result.feature_importance.empty:
            top_n = min(20, len(result.feature_importance))
            fi = result.feature_importance.head(top_n).reset_index()
            fi.columns = ["Feature", "Importance"]
            fig = px.bar(
                fi, x="Importance", y="Feature", orientation="h",
                title="Top Feature Importances",
                color="Importance", color_continuous_scale="Blues",
            )
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Feature importance not available for this model type.")

    # ── Raw Data tab ──────────────────────────────────────────────────────
    with tabs[2]:
        if not X.empty:
            st.write(
                f"The model used {len(X)} records and {X.shape[1]} features. "
                f"Most recent period in processed data: {X.index[-1] if len(X.index) else 'n/a'}."
            )
            with st.expander("Show processed feature table"):
                st.dataframe(X.tail(50), use_container_width=True)
        if raw_df is not None and not raw_df.empty:
            st.write(
                f"Raw source has {len(raw_df)} rows and {len(raw_df.columns)} columns."
            )
            with st.expander("Show raw source table"):
                st.dataframe(raw_df.tail(50), use_container_width=True)

    # ── Metrics tab ───────────────────────────────────────────────────────
    with tabs[3]:
        if result.metrics:
            metrics_df = pd.DataFrame.from_dict(
                result.metrics, orient="index", columns=["Value"]
            )
            st.table(metrics_df.style.format({"Value": "{:.4f}"}))
        else:
            st.info("No evaluation metrics available.")

    # ── Export tab ────────────────────────────────────────────────────────
    with tabs[4]:
        report = ReportBuilder(str(PROJECT_ROOT / "data" / "reports"))
        json_data = json.dumps(report.to_dict(result), indent=2, default=str)
        conf_for_export = _estimate_confidence(result, target_cfg.get("type", ""))
        decision_for_export = _decision_packet(
            target_cfg.get("id", ""), result, latest, conf_for_export, decision_policy
        )
        brief_txt = _single_run_narrative(
            target_cfg,
            result,
            conf_for_export,
            decision_for_export,
            X,
        )
        st.download_button(
            "⬇ Download JSON Report",
            data=json_data,
            file_name=f"{target_cfg['id']}_report.json",
            mime="application/json",
        )
        csv_data = result.predictions.to_csv()
        st.download_button(
            "⬇ Download Predictions CSV",
            data=csv_data,
            file_name=f"{target_cfg['id']}_predictions.csv",
            mime="text/csv",
        )
        st.download_button(
            "⬇ Download Natural Language Brief",
            data=brief_txt,
            file_name=f"{target_cfg['id']}_brief.txt",
            mime="text/plain",
        )


def _run_all_scaffold_targets(targets_cfg: dict, sources_cfg: dict, decision_policy: str = "Balanced"):
    st.subheader("Batch Run: All 4 Scaffold Targets")

    available = [k for k in SCAFFOLD_TARGET_KEYS if k in targets_cfg["targets"]]
    missing = [k for k in SCAFFOLD_TARGET_KEYS if k not in targets_cfg["targets"]]

    if missing:
        st.warning("Some scaffold targets are missing: " + ", ".join(missing))
    if not available:
        st.error("No scaffold targets found in configuration.")
        return

    rows = []
    combined_json = []
    report_builder = ReportBuilder(str(PROJECT_ROOT / "data" / "reports"))

    for key in available:
        cfg = dict(targets_cfg["targets"][key])
        cfg["id"] = key

        countries = cfg.get("countries", ["US"])
        start_year = int(cfg.get("date_range", {}).get("start", "2000")[:4])
        end_year = date.today().year
        horizon = int(cfg.get("forecast_horizon", 12)) if cfg.get("type") == "forecast" else 1

        try:
            result, X, _ = run_prediction(
                cfg, sources_cfg, countries, start_year, end_year, horizon, show_errors=False
            )
            if result is None:
                rows.append({
                    "target_key": key,
                    "target_label": cfg.get("label", key),
                    "status": "no_data",
                    "latest_prediction": None,
                    "model": None,
                    "metric_name": None,
                    "metric_value": None,
                })
                continue

            metric_name, metric_value = None, None
            if result.metrics:
                metric_name = next(iter(result.metrics))
                metric_value = result.metrics.get(metric_name)

            point_value = _latest_decision_value(result)
            confidence = _estimate_confidence(result, cfg.get("type", ""))
            decision = _decision_packet(key, result, point_value, confidence, decision_policy)

            rows.append({
                "target_key": key,
                "target_label": cfg.get("label", key),
                "status": "ok",
                "latest_prediction": point_value,
                "model": result.model_type,
                "metric_name": metric_name,
                "metric_value": metric_value,
                "feature_rows": 0 if X is None else len(X),
                "confidence": round(confidence, 3),
                "recommended_action": decision["action"],
                "urgency": decision["urgency"],
                "timeframe": decision["timeframe"],
            })

            combined_json.append({
                "target_key": key,
                "report": report_builder.to_dict(result),
            })
        except Exception as exc:
            rows.append({
                "target_key": key,
                "target_label": cfg.get("label", key),
                "status": "error",
                "latest_prediction": None,
                "model": None,
                "metric_name": None,
                "metric_value": None,
                "error": str(exc),
            })

    summary_df = pd.DataFrame(rows)
    _append_run_history(summary_df)
    st.session_state["batch_summary_csv"] = summary_df.to_csv(index=False)
    st.session_state["batch_detailed_json"] = json.dumps(combined_json, indent=2, default=str)
    st.session_state["batch_executive_text"] = _build_executive_summary(summary_df)
    st.session_state["batch_decision_memo_text"] = _build_decision_memo(summary_df)
    execution_df = _build_execution_plan(summary_df)
    st.session_state["batch_execution_plan_csv"] = execution_df.to_csv(index=False)
    st.session_state["batch_weekly_packet_text"] = _build_weekly_packet(
        summary_df,
        execution_df,
        _validation_snapshot(_load_outcomes()),
        decision_policy,
    )
    st.dataframe(summary_df, use_container_width=True)

    ok_count = int((summary_df["status"] == "ok").sum()) if not summary_df.empty else 0
    st.caption(f"Completed {ok_count}/{len(available)} targets successfully.")

    st.subheader("What This Means")
    if summary_df.empty:
        st.info("No outputs were generated.")
    else:
        for row in summary_df.to_dict(orient="records"):
            target_key = row.get("target_key", "")
            status = row.get("status", "")
            if status == "ok":
                prediction_text = _format_prediction_value(row.get("latest_prediction"))
                st.write(
                    f"- {row.get('target_label', target_key)}: {prediction_text}. "
                    f"Action: {row.get('recommended_action', 'review')}. "
                    f"Urgency: {row.get('urgency', 'medium')} ({row.get('timeframe', 'this week')}). "
                    f"Confidence: {float(row.get('confidence', 0)) * 100:.0f}%."
                )
            elif status == "no_data":
                st.write(f"- {row.get('target_label', target_key)}: no usable data found. Add/fill the matching template in data/uploads.")
            else:
                st.write(f"- {row.get('target_label', target_key)}: run failed. Check data columns in your uploaded template.")

    st.subheader("Batch Brief (Natural Language)")
    validation_now = _validation_snapshot(_load_outcomes())
    st.write(_batch_narrative(summary_df, validation_now))

    st.subheader("Decision Board")
    if not summary_df.empty:
        decision_cols = [
            "target_label", "latest_prediction", "confidence",
            "recommended_action", "urgency", "timeframe", "status"
        ]
        present_cols = [c for c in decision_cols if c in summary_df.columns]
        decision_view = summary_df[present_cols].copy()
        if "confidence" in decision_view.columns:
            decision_view["confidence"] = (
                decision_view["confidence"].fillna(0) * 100
            ).round(0).astype(int).astype(str) + "%"
        st.dataframe(decision_view, use_container_width=True)

    st.subheader("Action Queue (Next 7 Days)")
    if execution_df.empty:
        st.info("No actionable items yet.")
    else:
        st.dataframe(execution_df, use_container_width=True)

    st.subheader("Execution Tracker")
    tracker_seed = _seed_execution_tracker(execution_df)
    if tracker_seed.empty:
        st.info("No tracker rows yet.")
    else:
        editable_cols = [
            c for c in [
                "target_key", "target", "action", "urgency", "timeframe", "confidence",
                "owner", "due_date", "status", "next_step"
            ] if c in tracker_seed.columns
        ]
        tracker_view = tracker_seed[editable_cols].copy()
        tracker_edit = st.data_editor(
            tracker_view,
            use_container_width=True,
            num_rows="dynamic",
            key="execution_tracker_editor",
        )
        if st.button("Save Execution Tracker", use_container_width=True):
            _save_execution_tracker(pd.DataFrame(tracker_edit))
            st.success("Execution tracker saved.")

    st.subheader("Outcome Tracker")
    ok_rows = [r for r in summary_df.to_dict(orient="records") if r.get("status") == "ok"]
    if not ok_rows:
        st.info("Run successful targets first, then log outcomes here.")
    else:
        label_to_row = {r["target_label"]: r for r in ok_rows}
        target_label = st.selectbox("Target to log", list(label_to_row.keys()), key="outcome_target")
        selected = label_to_row[target_label]
        action_text = st.text_input(
            "Action taken",
            value=str(selected.get("recommended_action", "")),
            key="outcome_action",
        )
        outcome_grade = st.selectbox(
            "Result grade",
            ["hit", "partial", "miss"],
            key="outcome_grade",
            help="hit = outcome matched recommendation, partial = mixed, miss = did not work",
        )
        notes = st.text_area("Notes", value="", key="outcome_notes")
        if st.button("Save Outcome", use_container_width=True):
            _append_outcome_log(
                {
                    "logged_at": datetime.utcnow().isoformat(),
                    "target_key": selected.get("target_key"),
                    "target_label": selected.get("target_label"),
                    "signal": selected.get("latest_prediction"),
                    "confidence": selected.get("confidence"),
                    "action_taken": action_text,
                    "outcome_grade": outcome_grade,
                    "policy": decision_policy,
                    "notes": notes,
                }
            )
            st.success("Outcome logged.")

    outcomes_view = _load_outcomes(50)
    if not outcomes_view.empty:
        st.dataframe(outcomes_view.tail(15), use_container_width=True)

    snapshot = _validation_snapshot(outcomes_view)
    st.caption(
        f"4-week validation: {snapshot['hit_rate_4w'] * 100:.0f}% hit rate over {snapshot['rows_4w']} tracked outcomes."
    )

    st.download_button(
        "⬇ Download Combined Summary CSV",
        data=st.session_state["batch_summary_csv"],
        file_name="scaffold_batch_summary.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇ Download Combined Detailed JSON",
        data=st.session_state["batch_detailed_json"],
        file_name="scaffold_batch_report.json",
        mime="application/json",
    )

    st.download_button(
        "⬇ Download Executive Summary (Plain English)",
        data=st.session_state["batch_executive_text"],
        file_name="executive_summary.txt",
        mime="text/plain",
    )

    st.download_button(
        "⬇ Download Decision Memo (Actionable)",
        data=st.session_state["batch_decision_memo_text"],
        file_name="decision_memo.txt",
        mime="text/plain",
    )

    st.download_button(
        "⬇ Download Execution Plan (CSV)",
        data=st.session_state["batch_execution_plan_csv"],
        file_name="execution_plan.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇ Download Weekly Packet (Ready To Share)",
        data=st.session_state["batch_weekly_packet_text"],
        file_name="weekly_packet.txt",
        mime="text/plain",
    )


# ─── plot helpers ─────────────────────────────────────────────────────────────

def _plot_forecast(result):
    st.subheader("Historical Fit + Forecast")
    fitted = result.predictions.reset_index()
    fitted.columns = ["period", "value"]
    fitted["kind"] = "fitted"

    fc = result.forecast.reset_index()
    fc.columns = ["period", "forecast", "lower_95", "upper_95"]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=fitted["period"], y=fitted["value"],
        name="Historical fit", line=dict(color="#2563EB")
    ))
    fig.add_trace(go.Scatter(
        x=fc["period"], y=fc["forecast"],
        name="Forecast", line=dict(color="#16A34A", dash="dash")
    ))
    fig.add_trace(go.Scatter(
        x=pd.concat([fc["period"], fc["period"][::-1]]),
        y=pd.concat([fc["upper_95"], fc["lower_95"][::-1]]),
        fill="toself", fillcolor="rgba(22,163,74,0.1)",
        line=dict(color="rgba(22,163,74,0)"), name="95% CI"
    ))
    fig.update_layout(xaxis_title="Period", yaxis_title=result.target_label)
    st.plotly_chart(fig, use_container_width=True)


def _plot_probabilities(result):
    st.subheader("Class Probabilities Over Time")
    proba = result.probabilities.reset_index()
    date_col = proba.columns[0]
    proba_melted = proba.melt(id_vars=date_col, var_name="Class", value_name="Probability")
    fig = px.line(proba_melted, x=date_col, y="Probability", color="Class",
                  title="Predicted Class Probabilities")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Latest Prediction")
    latest_proba = result.probabilities.iloc[-1]
    fig2 = px.bar(
        x=latest_proba.index, y=latest_proba.values,
        labels={"x": "Class", "y": "Probability"},
        color=latest_proba.values, color_continuous_scale="Viridis",
    )
    st.plotly_chart(fig2, use_container_width=True)


def _plot_regression(result):
    st.subheader("Predictions Over Time")
    preds = result.predictions.reset_index()
    preds.columns = ["period", "prediction"]
    fig = px.line(preds, x="period", y="prediction",
                  title=f"{result.target_label} — Predicted Values",
                  labels={"prediction": result.target_label})
    st.plotly_chart(fig, use_container_width=True)


def _show_upload_panel():
    st.divider()
    st.subheader("📁 Upload Your Own Data")
    st.markdown(
        "Drop CSV, Excel, or Parquet files into `data/uploads/` — they will be "
        "automatically detected and merged with the feature pipeline."
    )
    with st.expander("Upload a file now"):
        uploaded = st.file_uploader(
            "Choose a file", type=["csv", "xlsx", "parquet"], accept_multiple_files=False
        )
        if uploaded:
            upload_dir = PROJECT_ROOT / "data" / "uploads"
            upload_dir.mkdir(parents=True, exist_ok=True)
            dest = upload_dir / uploaded.name
            dest.write_bytes(uploaded.read())
            st.success(f"Saved to {dest}")


def _format_prediction_value(value):
    if isinstance(value, float):
        return f"latest prediction is {value:.2f}"
    return f"latest prediction is {value}"


def _recommendation_for_target(target_key: str, prediction):
    text = str(prediction).lower() if prediction is not None else ""
    if target_key == "food_demand_weekly":
        return "Use this to set next-week inventory and staffing."
    if target_key == "construction_material_cost_direction":
        if "up" in text:
            return "Lock a larger share of supplier prices now and tighten cost controls."
        if "down" in text:
            return "Stagger purchases over coming weeks and avoid over-hedging."
        return "Keep procurement cadence stable and monitor weekly trend shifts."
    if target_key == "restaurant_failure_risk":
        if "high" in text:
            return "Prioritize intervention: pricing, labor, and rent controls."
        return "Maintain operations and track review + traffic trends."
    if target_key == "construction_project_delay_risk":
        if "major" in text:
            return "Escalate schedule mitigation now and secure backup subcontractors."
        return "Continue weekly schedule reviews and early blocker escalation."
    return "Review this output in your weekly planning cycle."


def _latest_decision_value(result):
    if result.forecast is not None and not result.forecast.empty and "forecast" in result.forecast.columns:
        return result.forecast.iloc[0]["forecast"]
    return result.latest_prediction()


def _estimate_confidence(result, target_type: str) -> float:
    if result.probabilities is not None and not result.probabilities.empty:
        return float(result.probabilities.iloc[-1].max())

    if target_type == "forecast" and result.forecast is not None and not result.forecast.empty:
        row = result.forecast.iloc[0]
        point = float(row.get("forecast", 0.0))
        lower = float(row.get("lower_95", point))
        upper = float(row.get("upper_95", point))
        width = max(0.0, upper - lower)
        relative = width / (abs(point) + 1e-6)
        return max(0.15, min(0.95, 1.0 - relative))

    if "accuracy" in result.metrics:
        return float(result.metrics["accuracy"])
    if "f1_weighted" in result.metrics:
        return float(result.metrics["f1_weighted"])
    if "R2" in result.metrics:
        r2 = float(result.metrics["R2"])
        return max(0.1, min(0.95, (r2 + 1.0) / 2.0))
    return 0.5


def _estimate_market_probability(target_key: str, result) -> float:
    """Return an estimated YES probability for prediction-market style decisions."""
    if result.probabilities is not None and not result.probabilities.empty:
        latest = result.probabilities.iloc[-1]
        lc = {str(k).lower(): float(v) for k, v in latest.items()}

        if target_key == "construction_material_cost_direction":
            if "up" in lc:
                return lc["up"]
        if target_key == "restaurant_failure_risk":
            if "high" in lc:
                return lc["high"]
            if "medium" in lc and "low" in lc:
                return min(0.99, lc["medium"] + lc["high"] if "high" in lc else lc["medium"])
        if target_key == "construction_project_delay_risk":
            if "major_delay" in lc:
                return lc["major_delay"]
            if "minor_delay" in lc:
                base = lc["minor_delay"]
                if "major_delay" in lc:
                    base += lc["major_delay"]
                return min(0.99, base)

        return float(max(latest.max(), 0.01))

    # For forecast-style outputs, convert confidence into a probability proxy.
    conf = _estimate_confidence(result, "forecast")
    return max(0.05, min(0.95, conf))


def _build_market_decision(
    model_prob: float,
    market_prob: float,
    bankroll: float,
    max_risk_pct: float,
) -> dict:
    edge = model_prob - market_prob
    no_trade_band = 0.05

    if abs(edge) < no_trade_band:
        return {
            "side": "NO TRADE",
            "stake_usd": 0.0,
            "edge_pct": edge * 100,
            "narrative": (
                "Your model and market are close. Edge is small, so best action is to wait "
                "for a better price rather than force a trade."
            ),
        }

    side = "BUY YES" if edge > 0 else "BUY NO"
    p = model_prob if side == "BUY YES" else 1.0 - model_prob
    kelly_raw = max(0.0, 2.0 * p - 1.0)
    kelly_fraction = 0.25 * kelly_raw  # conservative fractional Kelly
    risk_fraction = min(max_risk_pct, kelly_fraction if kelly_fraction > 0 else max_risk_pct * 0.25)
    stake = bankroll * risk_fraction

    narrative = (
        f"Suggested action: {side}. The model sees a {edge * 100:+.1f}% edge versus market pricing. "
        f"Cap stake near ${stake:.0f} ({risk_fraction * 100:.1f}% of bankroll) under your risk settings."
    )
    return {
        "side": side,
        "stake_usd": stake,
        "edge_pct": edge * 100,
        "narrative": narrative,
    }


def _decision_packet(
    target_key: str,
    result,
    latest_prediction,
    confidence: float,
    decision_policy: str = "Balanced",
) -> dict:
    action = _recommendation_for_target(target_key, latest_prediction)
    urgency = "medium"
    timeframe = "this week"

    policy_thresholds = {
        "Conservative": (0.82, 0.55),
        "Balanced": (0.75, 0.45),
        "Aggressive": (0.65, 0.35),
    }
    high_bar, low_bar = policy_thresholds.get(decision_policy, (0.75, 0.45))

    if confidence >= high_bar:
        urgency = "high"
        timeframe = "within 48 hours"
    elif confidence <= low_bar:
        urgency = "low"
        timeframe = "monitor and re-run next cycle"

    return {
        "action": action,
        "urgency": urgency,
        "timeframe": timeframe,
    }


def _build_execution_plan(summary_df: pd.DataFrame) -> pd.DataFrame:
    if summary_df.empty:
        return pd.DataFrame()

    urgency_rank = {"high": 3, "medium": 2, "low": 1}
    rows = []
    for row in summary_df.to_dict(orient="records"):
        if row.get("status") != "ok":
            continue
        urgency = str(row.get("urgency", "medium")).lower()
        confidence = float(row.get("confidence", 0.0))
        score = round(confidence * urgency_rank.get(urgency, 2), 3)
        rows.append(
            {
                "priority_score": score,
                "target": row.get("target_label"),
                "action": row.get("recommended_action"),
                "urgency": row.get("urgency"),
                "timeframe": row.get("timeframe"),
                "confidence": f"{confidence * 100:.0f}%",
                "owner": "TBD",
                "next_step": "Execute and review next run",
            }
        )

    if not rows:
        return pd.DataFrame()

    out = pd.DataFrame(rows).sort_values("priority_score", ascending=False)
    return out.reset_index(drop=True)


def _human_summary(target_key: str, latest_prediction, decision=None):
    pred = _format_prediction_value(latest_prediction)
    rec = _recommendation_for_target(target_key, latest_prediction)
    if decision is None:
        return f"Quick takeaway: {pred}. {rec}"
    return (
        f"Quick takeaway: {pred}. {rec} "
        f"Priority: {decision['urgency']} ({decision['timeframe']})."
    )


def _single_run_narrative(target_cfg: dict, result, confidence: float, decision: dict, X: pd.DataFrame) -> str:
    label = target_cfg.get("label", target_cfg.get("id", "Target"))
    latest = _latest_decision_value(result)
    quality_note = "Model quality metrics were not available for this run."
    if result.metrics:
        metric_name = next(iter(result.metrics))
        metric_val = result.metrics.get(metric_name)
        if isinstance(metric_val, float):
            quality_note = f"Primary quality signal is {metric_name}={metric_val:.3f}."
        else:
            quality_note = f"Primary quality signal is {metric_name}={metric_val}."

    return (
        f"Decision Narrative for {label}: The latest signal is {latest}. "
        f"Confidence is {confidence * 100:.0f}%. Recommended action: {decision['action']} "
        f"with {decision['urgency']} urgency ({decision['timeframe']}). "
        f"This run used {len(X)} rows and {X.shape[1] if not X.empty else 0} model features. "
        f"{quality_note}"
    )


def _batch_narrative(summary_df: pd.DataFrame, validation: dict) -> str:
    if summary_df.empty:
        return "No target outputs were generated in this batch."

    ok_count = int((summary_df["status"] == "ok").sum())
    lines = [
        f"Batch narrative: {ok_count}/{len(summary_df)} targets completed successfully.",
        f"Current deployment mode is {'Paper Mode' if validation.get('paper_only', True) else 'Limited Live Mode'}.",
        f"4-week tracked hit rate is {validation.get('hit_rate_4w', 0.0) * 100:.0f}% over {validation.get('rows_4w', 0)} outcomes.",
    ]

    for row in summary_df.to_dict(orient="records"):
        label = row.get("target_label", row.get("target_key", "target"))
        status = row.get("status", "unknown")
        if status != "ok":
            lines.append(f"{label}: {status}.")
            continue
        lines.append(
            f"{label}: signal={row.get('latest_prediction')}, confidence={float(row.get('confidence', 0))*100:.0f}%, "
            f"action={row.get('recommended_action')}, urgency={row.get('urgency')} ({row.get('timeframe')})."
        )
    return "\n".join(lines)


def _init_state():
    if "batch_summary_csv" not in st.session_state:
        st.session_state["batch_summary_csv"] = ""
    if "batch_detailed_json" not in st.session_state:
        st.session_state["batch_detailed_json"] = ""
    if "batch_executive_text" not in st.session_state:
        st.session_state["batch_executive_text"] = ""
    if "batch_decision_memo_text" not in st.session_state:
        st.session_state["batch_decision_memo_text"] = ""
    if "batch_execution_plan_csv" not in st.session_state:
        st.session_state["batch_execution_plan_csv"] = ""
    if "batch_weekly_packet_text" not in st.session_state:
        st.session_state["batch_weekly_packet_text"] = ""


def _build_executive_summary(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "No results were generated in the latest run."

    total = len(summary_df)
    ok_count = int((summary_df["status"] == "ok").sum())
    no_data_count = int((summary_df["status"] == "no_data").sum())
    error_count = int((summary_df["status"] == "error").sum())

    lines = [
        "Executive Summary",
        f"Completed runs: {ok_count}/{total}",
        f"No data: {no_data_count}",
        f"Errors: {error_count}",
        "",
        "Target Outcomes:",
    ]

    for row in summary_df.to_dict(orient="records"):
        label = row.get("target_label", row.get("target_key", "target"))
        status = row.get("status", "unknown")
        if status == "ok":
            pred = row.get("latest_prediction")
            lines.append(f"- {label}: {_format_prediction_value(pred)}. {_recommendation_for_target(row.get('target_key', ''), pred)}")
        elif status == "no_data":
            lines.append(f"- {label}: No usable data found. Fill and upload the matching template.")
        else:
            lines.append(f"- {label}: Run failed. Check column names and data completeness.")

    return "\n".join(lines)


def _build_decision_memo(summary_df: pd.DataFrame) -> str:
    if summary_df.empty:
        return "Decision Memo\n\nNo results available."

    lines = [
        "Decision Memo (Operational)",
        "Use this as a weekly action sheet for business, consulting, and market planning.",
        "",
    ]

    for row in summary_df.to_dict(orient="records"):
        label = row.get("target_label", row.get("target_key", "target"))
        status = row.get("status", "unknown")
        lines.append(label)
        lines.append(f"- Status: {status}")
        if status == "ok":
            lines.append(f"- Signal: {_format_prediction_value(row.get('latest_prediction'))}")
            lines.append(f"- Confidence: {float(row.get('confidence', 0)) * 100:.0f}%")
            lines.append(f"- Action: {row.get('recommended_action', 'review')}")
            lines.append(f"- Urgency: {row.get('urgency', 'medium')} ({row.get('timeframe', 'this week')})")
        elif status == "no_data":
            lines.append("- Action: Add/fill matching template file in data/uploads and rerun.")
        else:
            lines.append("- Action: Validate columns and missing values, then rerun.")
        lines.append("")

    lines.append("Note: This tool provides decision support, not legal/financial advice.")
    return "\n".join(lines)


def _build_weekly_packet(
    summary_df: pd.DataFrame,
    execution_df: pd.DataFrame,
    validation: dict,
    decision_policy: str,
) -> str:
    lines = [
        "Weekly Decision Packet",
        f"Generated at: {datetime.utcnow().isoformat()} UTC",
        f"Decision policy: {decision_policy}",
        "",
        "Readiness:",
        f"- Mode: {'Paper Mode' if validation.get('paper_only', True) else 'Limited Live Mode'}",
        f"- 4-week hit rate: {validation.get('hit_rate_4w', 0.0) * 100:.0f}%",
        f"- Tracked outcomes (4w): {validation.get('rows_4w', 0)}",
        f"- Gate note: {validation.get('note', '')}",
        "",
        "Signals:",
    ]

    if summary_df.empty:
        lines.append("- No signals generated this cycle.")
    else:
        for row in summary_df.to_dict(orient="records"):
            label = row.get("target_label", row.get("target_key", "target"))
            status = row.get("status", "unknown")
            if status != "ok":
                lines.append(f"- {label}: {status}")
                continue
            lines.append(
                f"- {label}: {_format_prediction_value(row.get('latest_prediction'))}; "
                f"confidence {float(row.get('confidence', 0)) * 100:.0f}%; "
                f"action {row.get('recommended_action', 'review')}; "
                f"urgency {row.get('urgency', 'medium')} ({row.get('timeframe', 'this week')})"
            )

    lines.append("")
    lines.append("Execution Queue:")
    if execution_df.empty:
        lines.append("- No executable items.")
    else:
        for row in execution_df.to_dict(orient="records"):
            lines.append(
                f"- {row.get('target')}: {row.get('action')} | "
                f"urgency {row.get('urgency')} | timeframe {row.get('timeframe')} | "
                f"confidence {row.get('confidence')}"
            )

    lines.append("")
    lines.append("Disclaimer: Decision support only; not legal or financial advice.")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
