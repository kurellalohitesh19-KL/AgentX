"""SpendWise AI - Expense Auditor & Anomaly Hunter.

Interactive Fintech Web Application for the Finance & Risk Track (AgentX Challenge 2026).
Seamlessly connects to the underlying ReAct agent and 5 specialized financial analytics tools.
"""

from __future__ import annotations

import io
import json
import os
import sys
import tempfile
from decimal import Decimal, ROUND_HALF_UP
from typing import Any, Dict, List, Optional

import altair as alt
import numpy as np
import pandas as pd
import streamlit as st

# Ensure workspace root is in sys.path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import config
from main import run_agent
from tools.categorizer import Categorizer
from tools.duplicate_detector import DuplicateDetector
from tools.anomaly_detector import AnomalyDetector
from tools.subscription_detector import SubscriptionDetector
from tools.budget_planner import BudgetPlanner
from tools.report_generator import FinancialReportGenerator
from tools.transaction_loader import (
    load_transactions,
    normalize_transactions,
    resolve_data_path,
    validate_transactions,
)

# ==============================================================================
# FINTECH THEME STYLESHEET
# ==============================================================================
CUSTOM_CSS = """
<style>
/* Main Dark Navy & Emerald Fintech Palette */
:root {
    --primary-bg: #0B132B;
    --card-bg: #1C2541;
    --accent-emerald: #10B981;
    --accent-teal: #34D399;
    --text-main: #F8FAFC;
    --text-muted: #94A3B8;
    --border-subtle: #2D3748;
}

/* Metric Card Customization */
div[data-testid="stMetric"] {
    background-color: #1C2541;
    border: 1px solid #2D3748;
    border-radius: 10px;
    padding: 14px 18px;
    box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
}

div[data-testid="stMetricLabel"] p {
    color: #94A3B8 !important;
    font-size: 0.85rem !important;
    font-weight: 600 !important;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

div[data-testid="stMetricValue"] {
    color: #10B981 !important;
    font-weight: 700 !important;
}

/* Pill Badges */
.badge-danger {
    background-color: rgba(239, 68, 68, 0.15);
    color: #EF4444;
    border: 1px solid rgba(239, 68, 68, 0.3);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-warning {
    background-color: rgba(245, 158, 11, 0.15);
    color: #F59E0B;
    border: 1px solid rgba(245, 158, 11, 0.3);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}
.badge-success {
    background-color: rgba(16, 185, 129, 0.15);
    color: #10B981;
    border: 1px solid rgba(16, 185, 129, 0.3);
    padding: 3px 8px;
    border-radius: 6px;
    font-size: 0.78rem;
    font-weight: 600;
}

/* Table Clean Styling */
div[data-testid="stDataFrame"] {
    border: 1px solid #2D3748;
    border-radius: 8px;
}

/* Sidebar Branding */
.sidebar-header {
    padding: 10px 0 20px 0;
    text-align: left;
}
.sidebar-title {
    font-size: 1.25rem;
    font-weight: 800;
    color: #10B981;
    margin-bottom: 2px;
}
.sidebar-subtitle {
    font-size: 0.75rem;
    color: #94A3B8;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}
</style>
"""


# ==============================================================================
# DATA CACHING & BACKEND WRAPPERS
# ==============================================================================
@st.cache_data(show_spinner=False)
def get_dataset(filepath: str) -> pd.DataFrame:
    return load_transactions(filepath)


@st.cache_data(show_spinner=False)
def get_cached_audit(
    filepath: str,
    monthly_income: Optional[float] = None,
    fixed_commitments: float = 0.0,
    savings_goal_pct: float = 20.0,
) -> Dict[str, Any]:
    generator = FinancialReportGenerator(filename=filepath)
    return generator.generate_full_audit(
        monthly_income=monthly_income,
        fixed_commitments=fixed_commitments,
        savings_goal_pct=savings_goal_pct,
    )


def resolve_app_data_path(preferred_name: str = "transactions.csv") -> str:
    """Safely resolves an active transaction dataset path with cascading fallbacks."""
    try:
        return resolve_data_path(preferred_name)
    except FileNotFoundError:
        base_dir = os.environ.get("AGENTX_DATA_DIR") or config.DATA_DIR
        if os.path.exists(base_dir):
            csvs = [f for f in os.listdir(base_dir) if f.endswith(".csv")]
            if csvs:
                return os.path.join(base_dir, csvs[0])
        synthetic_path = os.path.join(BASE_DIR, "tests", "data", "synthetic_transactions.csv")
        if os.path.exists(synthetic_path):
            return synthetic_path
        return os.path.join(BASE_DIR, "data", "transactions.csv")


# ==============================================================================
# DASHBOARD APPLICATION CONTROLLER
# ==============================================================================
def run_dashboard():
    st.set_page_config(
        page_title="SpendWise AI — Expense Auditor & Anomaly Hunter",
        page_icon="🛡️",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

    # --------------------------------------------------------------------------
    # SIDEBAR
    # --------------------------------------------------------------------------
    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-header">
                <div class="sidebar-title">🛡️ SpendWise AI</div>
                <div class="sidebar-subtitle">Expense Auditor & Anomaly Hunter</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        st.subheader("Data Source")
        data_source = st.radio(
            "Select active dataset:",
            [
                "📦 Kaggle UPI 2024 (250k)",
                "🧪 Synthetic Fixture (14)",
                "📁 Upload Custom CSV",
            ],
            index=0,
        )

        active_filepath = resolve_app_data_path("transactions.csv")
        uploaded_name = None

        if data_source == "📦 Kaggle UPI 2024 (250k)":
            active_filepath = resolve_app_data_path("transactions.csv")
        elif data_source == "🧪 Synthetic Fixture (14)":
            synthetic_cand = os.path.join(BASE_DIR, "tests", "data", "synthetic_transactions.csv")
            if os.path.exists(synthetic_cand):
                active_filepath = synthetic_cand
            else:
                active_filepath = resolve_app_data_path("transactions.csv")
        elif data_source == "📁 Upload Custom CSV":
            uploaded_file = st.file_uploader("Upload transactions CSV", type=["csv"])
            if uploaded_file is not None:
                temp_dir = os.path.join(BASE_DIR, "data", "temp")
                os.makedirs(temp_dir, exist_ok=True)
                temp_path = os.path.join(temp_dir, f"upload_{uploaded_file.name}")
                with open(temp_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                active_filepath = temp_path
                uploaded_name = uploaded_file.name
                st.success(f"Loaded: `{uploaded_name}`")
            else:
                st.info("Upload a CSV file to inspect your own transactions.")
                active_filepath = resolve_app_data_path("transactions.csv")

        st.markdown("---")
        st.subheader("Navigation")
        page = st.selectbox(
            "Go to page:",
            [
                "1. 📊 Executive Dashboard",
                "2. 🔍 Transaction Explorer",
                "3. 📈 Expense Analytics",
                "4. 🚨 Anomaly Center",
                "5. 🔄 Subscription Watch",
                "6. 💡 Smart Budget Planner",
                "7. 🤖 AI Financial Auditor",
                "8. 📑 Financial Reports",
            ],
            index=0,
        )

        st.markdown("---")
        has_gemini = bool(os.environ.get("GEMINI_API_KEY"))
        has_groq = bool(os.environ.get("GROQ_API_KEY"))
        active_provider = (
            "Gemini" if has_gemini else "Groq" if has_groq else "DryRun (Offline)"
        )
        st.caption(f"**Agent Provider:** `{active_provider}`")
        st.caption(f"**Max Iterations:** `{config.MAX_ITERATIONS}`")
        st.caption("**Challenge Track:** Finance & Risk AgentX 2026")

    # Load Active Dataset
    try:
        df_active = get_dataset(active_filepath)
        audit_data = get_cached_audit(active_filepath)
    except Exception as e:
        st.error(f"Error loading dataset: {e}")
        return

    # --------------------------------------------------------------------------
    # PAGE 1: 📊 EXECUTIVE DASHBOARD
    # --------------------------------------------------------------------------
    if page.startswith("1."):
        st.title("📊 Financial Executive Dashboard")
        st.markdown(
            f"**Active Portfolio Dataset:** `{os.path.basename(active_filepath)}` — "
            f"**Records Audited:** `{len(df_active):,}`"
        )

        col1, col2, col3, col4, col5 = st.columns(5)
        total_spend = audit_data["categorization"].get("total_spending", 0.0)
        leak_est = audit_data.get("recoverable_leak_estimate", 0.0)
        anom_count = audit_data.get("anomalies_detected", 0)
        health_score = audit_data.get("health_score", 75)

        col1.metric("Total Spending", f"₹{total_spend:,.0f}")
        col2.metric("Total Transactions", f"{len(df_active):,}")
        col3.metric("Recoverable Leaks", f"₹{leak_est:,.2f}")
        col4.metric("Spending Anomalies", f"{anom_count:,}")
        col5.metric("Health Score", f"{health_score}/100")

        st.markdown("---")
        row1_col1, row1_col2 = st.columns([1, 1])

        with row1_col1:
            st.subheader("Top Spending Categories")
            cats = audit_data["categorization"].get("category_breakdown", [])
            if cats:
                df_cats = pd.DataFrame(cats)
                chart = (
                    alt.Chart(df_cats.head(8))
                    .mark_bar(cornerRadiusTopRight=4, cornerRadiusBottomRight=4)
                    .encode(
                        x=alt.X("total_spend:Q", title="Total Spend (INR)"),
                        y=alt.Y("category:N", sort="-x", title="Category"),
                        color=alt.value("#10B981"),
                        tooltip=["category", "total_spend", "percent_of_total_spend"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(chart, use_container_width=True)

        with row1_col2:
            st.subheader("Monthly Spending Run-Rate")
            if "timestamp" in df_active.columns:
                df_ts = df_active.dropna(subset=["timestamp", "amount"]).copy()
                df_ts["month"] = pd.to_datetime(df_ts["timestamp"], errors="coerce").dt.to_period("M").astype(str)
                monthly_trend = df_ts.groupby("month")["amount"].sum().reset_index()

                trend_chart = (
                    alt.Chart(monthly_trend)
                    .mark_area(
                        line={"color": "#10B981"},
                        color=alt.Gradient(
                            gradient="linear",
                            stops=[
                                alt.GradientStop(color="#10B981", offset=0),
                                alt.GradientStop(color="rgba(16, 185, 129, 0.05)", offset=1),
                            ],
                            x1=1,
                            x2=1,
                            y1=1,
                            y2=0,
                        ),
                    )
                    .encode(
                        x=alt.X("month:N", title="Month"),
                        y=alt.Y("amount:Q", title="Monthly Spend (INR)"),
                        tooltip=["month", "amount"],
                    )
                    .properties(height=320)
                )
                st.altair_chart(trend_chart, use_container_width=True)

        st.subheader("Audit Alerts & Recommended Actions")
        f_col1, f_col2, f_col3 = st.columns(3)

        with f_col1:
            st.markdown(
                f"""
                <div style="background:#1C2541; padding:16px; border-radius:8px; border:1px solid #2D3748;">
                    <div class="badge-danger">DUPLICATE PAYMENTS</div>
                    <h4 style="margin:10px 0 5px 0;">₹{leak_est:,.2f} Flagged Leaks</h4>
                    <p style="color:#94A3B8; font-size:0.85rem;">
                        Identified {audit_data.get('potential_duplicates', 0)} possible duplicate payments
                        within short time windows. Review in Anomaly Center to request refunds.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with f_col2:
            st.markdown(
                f"""
                <div style="background:#1C2541; padding:16px; border-radius:8px; border:1px solid #2D3748;">
                    <div class="badge-warning">STATISTICAL OUTLIERS</div>
                    <h4 style="margin:10px 0 5px 0;">{anom_count:,} Anomalous Spikes</h4>
                    <p style="color:#94A3B8; font-size:0.85rem;">
                        Transactions exceeding category baseline IQR fences (Q3 + 1.5*IQR).
                        No synthetic fraud labels used; pure behavioral deviation.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

        with f_col3:
            annual_sub = audit_data.get("annual_subscription_commitment", 0.0)
            st.markdown(
                f"""
                <div style="background:#1C2541; padding:16px; border-radius:8px; border:1px solid #2D3748;">
                    <div class="badge-success">RECURRING COMMITMENTS</div>
                    <h4 style="margin:10px 0 5px 0;">₹{annual_sub:,.2f} / Year</h4>
                    <p style="color:#94A3B8; font-size:0.85rem;">
                        Projected annual recurring expense commitment across verified monthly and weekly cadences.
                    </p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    # --------------------------------------------------------------------------
    # PAGE 2: 🔍 TRANSACTION EXPLORER
    # --------------------------------------------------------------------------
    elif page.startswith("2."):
        st.title("🔍 Transaction Explorer & Data Filter")
        st.markdown("Search, filter, and inspect canonicalized transaction records.")

        f_col1, f_col2, f_col3, f_col4 = st.columns(4)
        with f_col1:
            search_id = st.text_input("Search Transaction ID", "").strip()
        with f_col2:
            categories = ["All"] + sorted(list(df_active["category"].unique()))
            selected_cat = st.selectbox("Category Filter", categories)
        with f_col3:
            statuses = ["All"] + sorted(list(df_active["status"].unique())) if "status" in df_active.columns else ["All"]
            selected_status = st.selectbox("Status Filter", statuses)
        with f_col4:
            min_a = float(df_active["amount"].min()) if not df_active.empty else 0.0
            max_a = float(df_active["amount"].max()) if not df_active.empty else 10000.0
            amt_filter = st.slider("Max Amount (INR)", min_value=min_a, max_value=max_a, value=max_a)

        filtered = df_active.copy()
        if search_id:
            filtered = filtered[filtered["transaction_id"].str.contains(search_id, case=False, na=False)]
        if selected_cat != "All":
            filtered = filtered[filtered["category"] == selected_cat]
        if selected_status != "All" and "status" in filtered.columns:
            filtered = filtered[filtered["status"] == selected_status]
        filtered = filtered[filtered["amount"] <= amt_filter]

        st.write(f"Showing **{len(filtered):,}** matching transactions:")
        display_cols = [c for c in ["transaction_id", "timestamp", "amount", "category", "status", "sender_bank", "receiver_bank", "transaction_type"] if c in filtered.columns]
        st.dataframe(filtered[display_cols].head(100), use_container_width=True)

        if search_id and not filtered.empty:
            st.subheader(f"Transaction Detail: {filtered.iloc[0]['transaction_id']}")
            st.json(filtered.iloc[0].to_dict())

    # --------------------------------------------------------------------------
    # PAGE 3: 📈 EXPENSE ANALYTICS
    # --------------------------------------------------------------------------
    elif page.startswith("3."):
        st.title("📈 Expense Analytics & Category Breakdown")
        cats = audit_data["categorization"].get("category_breakdown", [])
        if cats:
            df_c = pd.DataFrame(cats)
            st.dataframe(
                df_c.style.format({
                    "total_spend": "₹{:,.2f}",
                    "average_spend": "₹{:,.2f}",
                    "percent_of_total_spend": "{:.2f}%",
                }),
                use_container_width=True,
            )
            st.subheader("Category Expenditure Comparison")
            chart = (
                alt.Chart(df_c)
                .mark_bar()
                .encode(
                    x=alt.X("category:N", sort="-y", title="Category"),
                    y=alt.Y("total_spend:Q", title="Total Expenditure (INR)"),
                    color=alt.Color("category:N", legend=None),
                    tooltip=["category", "total_spend", "transaction_count"],
                )
                .properties(height=380)
            )
            st.altair_chart(chart, use_container_width=True)

        if "timestamp" in df_active.columns:
            st.subheader("Daily Spending Activity (Recent 90 Days)")
            df_d = df_active.dropna(subset=["timestamp", "amount"]).copy()
            df_d["date"] = pd.to_datetime(df_d["timestamp"], errors="coerce").dt.date
            daily_sum = df_d.groupby("date")["amount"].sum().reset_index().tail(90)
            d_chart = (
                alt.Chart(daily_sum)
                .mark_line(color="#10B981")
                .encode(
                    x=alt.X("date:T", title="Date"),
                    y=alt.Y("amount:Q", title="Daily Spending (INR)"),
                    tooltip=["date", "amount"],
                )
                .properties(height=300)
            )
            st.altair_chart(d_chart, use_container_width=True)

    # --------------------------------------------------------------------------
    # PAGE 4: 🚨 ANOMALY CENTER
    # --------------------------------------------------------------------------
    elif page.startswith("4."):
        st.title("🚨 Anomaly Center & Double-Billing Hunter")
        st.markdown("Grounds every finding in explainable statistical IQR fences and Z-scores.")

        tab_dup, tab_outlier, tab_surges = st.tabs([
            "👯 Duplicate Payments",
            "📈 Statistical Outliers (IQR / Z-Score)",
            "🌊 Category Surges",
        ])

        with tab_dup:
            st.subheader("Identified Duplicate Payments & Double Debits")
            dups = audit_data["duplicates"].get("potential_duplicate_payments", [])
            if dups:
                dup_rows = []
                for d in dups:
                    dup_rows.append({
                        "Risk Level": d["risk_level"],
                        "Amount (INR)": f"₹{d['amount']:,.2f}",
                        "Interval (min)": d["time_interval_minutes"],
                        "First Txn": d["first_transaction"]["transaction_id"],
                        "First Status": d["first_transaction"]["status"],
                        "Second Txn": d["second_transaction"]["transaction_id"],
                        "Second Status": d["second_transaction"]["status"],
                        "Recoverable Leak": f"₹{d['potential_recoverable_leak']:,.2f}",
                        "Explanation": d["explanation"],
                    })
                st.dataframe(pd.DataFrame(dup_rows), use_container_width=True)
            else:
                st.info("No suspicious duplicate payments detected.")

        with tab_outlier:
            st.subheader("Unusually Large Transactions (Statistical Outliers)")
            anoms = audit_data["anomalies"].get("anomalous_transactions", [])
            if anoms:
                anom_rows = []
                for a in anoms:
                    anom_rows.append({
                        "Txn ID": a["transaction_id"],
                        "Date": a["timestamp"],
                        "Category": a["category"],
                        "Amount (INR)": f"₹{a['amount']:,.2f}",
                        "Category Median": f"₹{a['category_median']:,.2f}",
                        "Ratio to Median": a["ratio_to_median"],
                        "Z-Score": a["z_score"],
                        "Severity": a["severity"],
                        "Explanation": a["explanation"],
                    })
                st.dataframe(pd.DataFrame(anom_rows), use_container_width=True)
            else:
                st.info("No extreme statistical outliers detected.")

        with tab_surges:
            st.subheader("Category Monthly Spending Surges")
            surges = audit_data["anomalies"].get("category_spending_surges", [])
            if surges:
                st.dataframe(pd.DataFrame(surges), use_container_width=True)
            else:
                st.info("No periodic category spending surges exceeding +40% threshold.")

    # --------------------------------------------------------------------------
    # PAGE 5: 🔄 SUBSCRIPTION WATCH
    # --------------------------------------------------------------------------
    elif page.startswith("5."):
        st.title("🔄 Subscription Watch & Price Creep Tracker")
        subs = audit_data["subscriptions"].get("confirmed_subscriptions", [])
        missing_fields = audit_data["subscriptions"].get("summary", {}).get("missing_fields_identified", [])

        if "merchant" in missing_fields:
            st.warning(
                "⚠️ Notice: Specific merchant name column is absent in this dataset. Recurring groupings "
                "are established via (category, payment_type, destination_bank) cadences without fabricating merchant identities."
            )

        if subs:
            sub_rows = []
            for s in subs:
                creep = "None"
                if s.get("price_creep"):
                    creep = f"+{s['price_creep']['increase_per_cycle']} ({s['price_creep']['percentage_increase']})"
                sub_rows.append({
                    "Recurring Group": str(s["entity"]),
                    "Cadence": s["cadence"],
                    "Cycles Observed": s["cycle_count"],
                    "Mean Interval (Days)": s["mean_interval_days"],
                    "Latest Amount": f"₹{s['latest_amount']:,.2f}",
                    "Est. Annual Cost": f"₹{s['estimated_annual_cost']:,.2f}",
                    "Price Creep": creep,
                    "First Observed": s["first_transaction_date"],
                    "Latest Observed": s["latest_transaction_date"],
                })
            st.dataframe(pd.DataFrame(sub_rows), use_container_width=True)
        else:
            st.info("No regular periodic subscriptions identified at current cadence tolerance.")

    # --------------------------------------------------------------------------
    # PAGE 6: 💡 SMART BUDGET PLANNER
    # --------------------------------------------------------------------------
    elif page.startswith("6."):
        st.title("💡 Smart 50/30/20 Budget Planner & Optimizer")
        b_col1, b_col2, b_col3 = st.columns(3)
        with b_col1:
            income_input = st.number_input("Monthly Net Income (INR)", min_value=0.0, value=75000.0, step=5000.0)
        with b_col2:
            commit_input = st.number_input("Fixed Commitments / EMIs (INR)", min_value=0.0, value=15000.0, step=1000.0)
        with b_col3:
            savings_goal = st.slider("Target Savings Rate (%)", min_value=5.0, max_value=50.0, value=20.0, step=5.0)

        bud_tool = BudgetPlanner()
        bud_result = json.loads(bud_tool.execute(
            filename=active_filepath,
            monthly_income=income_input,
            fixed_commitments=commit_input,
            savings_goal_pct=savings_goal,
        ))

        if bud_result.get("status") == "success":
            score_val = bud_result["financial_health_score"]["score"]
            rating = bud_result["financial_health_score"]["rating"]
            st.metric("Financial Health Score", f"{score_val}/100", f"Rating: {rating}")
            st.info(f"**Recommendation Summary:** {bud_result['recommendation_summary']}")

            fw = bud_result["budget_framework_50_30_20"]
            c1, c2, c3 = st.columns(3)
            c1.metric("Needs (50% Target)", f"₹{fw['needs_50']['actual_monthly_spending']:,.2f}", f"Target: ₹{fw['needs_50']['target_allocation']:,.2f}")
            c2.metric("Wants (30% Target)", f"₹{fw['wants_30']['actual_monthly_spending']:,.2f}", f"Target: ₹{fw['wants_30']['target_allocation']:,.2f}")
            c3.metric("Savings Net Surplus", f"₹{fw['savings_20']['actual_monthly_savings']:,.2f}", f"Target: ₹{fw['savings_20']['target_monthly_savings']:,.2f}")

            st.subheader("Proposed Category Spending Caps")
            caps = bud_result.get("proposed_category_limits", {})
            if caps:
                cap_rows = []
                for cat, details in caps.items():
                    cap_rows.append({
                        "Category": cat,
                        "Current Monthly Spend": f"₹{details['current_monthly_spend']:,.2f}",
                        "Proposed Monthly Cap": f"₹{details['proposed_monthly_cap']:,.2f}",
                        "Potential Monthly Savings": f"₹{details['potential_monthly_savings']:,.2f}",
                        "Status": details["status"],
                    })
                st.dataframe(pd.DataFrame(cap_rows), use_container_width=True)

    # --------------------------------------------------------------------------
    # PAGE 7: 🤖 AI FINANCIAL AUDITOR (ReAct Chat)
    # --------------------------------------------------------------------------
    elif page.startswith("7."):
        st.title("🤖 AI Financial Auditor (ReAct Agent)")
        st.markdown("Chat directly with the autonomous SpendWise ReAct Agent. Inspect thoughts, tool calls, and observations.")

        if "messages" not in st.session_state:
            st.session_state["messages"] = [
                {
                    "role": "assistant",
                    "content": (
                        "Hello! I am **SpendWise AI**, your autonomous financial auditor. "
                        "I can audit transactions, categorize expenses, detect duplicate payments, "
                        "hunt for anomalies, and optimize your 50/30/20 budget. "
                        "How can I audit your finances today?"
                    ),
                }
            ]

        st.write("**Quick Audit Queries:**")
        q_col1, q_col2, q_col3, q_col4 = st.columns(4)
        preset_query = None
        if q_col1.button("📊 Audit Spending"):
            preset_query = "Run a financial audit on my transactions and categorize expenses."
        if q_col2.button("👯 Check Duplicates"):
            preset_query = "Did I get double-charged or are there any duplicate payments?"
        if q_col3.button("🚨 Find Anomalies"):
            preset_query = "Find statistical spending anomalies and surges in my transactions."
        if q_col4.button("💡 50/30/20 Budget"):
            preset_query = "Plan a 50/30/20 budget for an income of 75000 and fixed commitments of 15000."

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                if "trace" in msg and msg["trace"]:
                    with st.expander("🔍 ReAct Reasoning Trace (Thoughts & Tools)"):
                        for step_idx, step in enumerate(msg["trace"], 1):
                            st.markdown(f"**Step {step_idx}:** Tool `{step['tool']}`")
                            st.caption(f"*Thought:* {step['thought']}")
                            st.code(json.dumps(step.get("args", {}), indent=2), language="json")
                            st.text_area(f"Observation {step_idx}", step["observation"][:1000], height=100)

        user_query = st.chat_input("Ask a financial audit question...")
        effective_query = preset_query or user_query

        if effective_query:
            st.session_state["messages"].append({"role": "user", "content": effective_query})
            with st.chat_message("user"):
                st.markdown(effective_query)

            with st.chat_message("assistant"):
                with st.spinner("SpendWise AI is inspecting transaction logs and executing tools..."):
                    try:
                        dry_run = not (bool(os.environ.get("GEMINI_API_KEY")) or bool(os.environ.get("GROQ_API_KEY")))
                        answer, trace = run_agent(effective_query, dry_run=dry_run, verbose=False)
                    except Exception as e:
                        answer = f"Agent error: {type(e).__name__}: {e}"
                        trace = []

                    st.markdown(answer)

                    if trace:
                        with st.expander("🔍 ReAct Reasoning Trace (Thoughts & Tools)"):
                            for step_idx, step in enumerate(trace, 1):
                                st.markdown(f"**Step {step_idx}:** Tool `{step['tool']}`")
                                st.caption(f"*Thought:* {step['thought']}")
                                st.code(json.dumps(step.get("args", {}), indent=2), language="json")
                                st.text_area(f"Observation {step_idx}", step["observation"][:1000], height=100)

            st.session_state["messages"].append({
                "role": "assistant",
                "content": answer,
                "trace": trace,
            })

    # --------------------------------------------------------------------------
    # PAGE 8: 📑 FINANCIAL REPORTS
    # --------------------------------------------------------------------------
    elif page.startswith("8."):
        st.title("📑 Financial Audit Reports & Export")
        st.markdown("Generate and export executive-ready audit reports and structured CSV data.")

        generator = FinancialReportGenerator(filename=active_filepath)
        md_report = generator.export_markdown_report(audit_data)
        csv_data = generator.export_csv_summary(audit_data)

        d_col1, d_col2 = st.columns(2)
        with d_col1:
            st.download_button(
                label="📥 Download Audit Report (.md)",
                data=md_report,
                file_name=f"spendwise_audit_report_{os.path.basename(active_filepath)}.md",
                mime="text/markdown",
            )
        with d_col2:
            st.download_button(
                label="📥 Download Spending Summary (.csv)",
                data=csv_data,
                file_name="spendwise_spending_summary.csv",
                mime="text/csv",
            )

        st.markdown("---")
        st.subheader("Report Preview")
        st.markdown(md_report)


if __name__ == "__main__":
    run_dashboard()
