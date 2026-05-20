"""Converted from crewai_roi_improved_campaign_planner.ipynb.

Usage:
    python campaign_roi_crewai.py --input customer_behavior_with_reviews_scores.csv --output-dir outputs
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import date, datetime
from pathlib import Path

import numpy as np
import pandas as pd
from pulp import LpBinary, LpMaximize, LpProblem, LpVariable, lpSum
import os
from dotenv import load_dotenv

import streamlit as st



load_dotenv()

def get_openai_api_key():
    # First try environment variable
    api_key = os.getenv("OPENAI_API_KEY")

    # Then try Streamlit secrets only if available
    try:
        api_key = api_key or st.secrets["OPENAI_API_KEY"]
    except Exception:
        pass

    return api_key

OPENAI_API_KEY = get_openai_api_key()

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
else:
    st.warning("OPENAI_API_KEY not found. CrewAI text output will be skipped.")

#OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", os.getenv("OPENAI_API_KEY"))

try:
    from crewai import Agent, Crew, Process, Task
    CREWAI_AVAILABLE = True
except Exception:
    CREWAI_AVAILABLE = False
    
print("CREWAI AVAILABLE:", CREWAI_AVAILABLE)
print("OPENAI KEY FOUND:", bool(OPENAI_API_KEY))

pd.set_option("display.max_columns", 200)
pd.set_option("display.float_format", lambda x: f"{x:,.4f}")

profit_margin_rate = 0.22
campaign_cost_rate = 0.05
fixed_monthly_campaign_cost = 25000
quarterly_budget = 180000
max_selected_months = 1
max_resource_utilization = 0.85




# Explicitly load your .env file
load_dotenv(r"C:\Users\Reet\Downloads\MLOPS\campaign\.env")

# Optional debug (remove later)
if not os.getenv("OPENAI_API_KEY"):
    raise ValueError("OPENAI_API_KEY not found after loading .env")


def add_roi_signals(df: pd.DataFrame) -> pd.DataFrame:
    temp = df.copy()
    temp["expected_base_revenue"] = temp["online_retail_value"] * temp["buy_prob"] * temp["spend_propensity_score"]
    temp["expected_uplift_revenue"] = temp["expected_base_revenue"] * temp["spend_uplift_score"]
    temp["expected_total_revenue"] = (temp["expected_base_revenue"] + temp["expected_uplift_revenue"]) * (1 - temp["return_prob"])
    temp["expected_incremental_profit"] = temp["expected_total_revenue"] * profit_margin_rate
    temp["estimated_campaign_cost"] = temp["online_retail_value"] * campaign_cost_rate
    temp["expected_roi"] = np.where(
        temp["estimated_campaign_cost"] > 0,
        temp["expected_incremental_profit"] / temp["estimated_campaign_cost"],
        0,
    )

    conditions = [
        (temp["expected_roi"] >= 3.0) & (temp["return_prob"] < 0.20),
        (temp["expected_roi"] >= 1.5) & (temp["return_prob"] < 0.35),
        (temp["expected_roi"] >= 0.75),
    ]
    choices = ["High", "Medium", "Low"]
    temp["commercial_priority_band"] = np.select(conditions, choices, default="Deprioritize")
    temp["predicted_future_value_summary"] = np.select(
        [
            temp["commercial_priority_band"].eq("High"),
            temp["commercial_priority_band"].eq("Medium"),
            temp["commercial_priority_band"].eq("Low"),
        ],
        [
            "High-value target with strong expected ROI and manageable return risk.",
            "Promising target with positive ROI; suitable for selective campaign investment.",
            "Marginal target with limited upside; use lower-cost or controlled outreach.",
        ],
        default="Low commercial attractiveness; avoid allocating primary campaign budget.",
    )
    return temp


def sanitize_for_crewai(obj):
    if isinstance(obj, dict):
        return {str(k): sanitize_for_crewai(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_for_crewai(v) for v in obj]
    if isinstance(obj, tuple):
        return [sanitize_for_crewai(v) for v in obj]
    if isinstance(obj, pd.Timestamp):
        return obj.isoformat()
    if isinstance(obj, (datetime, date)):
        return obj.isoformat()
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    if pd.isna(obj):
        return None
    return obj


def build_crewai_artifacts(top_targets_df: pd.DataFrame, optimized_plan_df: pd.DataFrame):
    solution_analyst_prompt_template = """
You are the Solution Analyst.

You are given sample high-value customer records:
{top_targets_sample}

Your job:
1. Analyze expected ROI, expected incremental profit, recommendation probability,
   return risk, and spend uplift.
2. Explain what kind of customers should be prioritized for the next quarter campaign.
3. Do NOT use customer reviews.
4. Produce a structured commercial opportunity summary.

Return STRICT JSON:
{
  "priority_customer_profile": "...",
  "key_value_drivers": ["...", "...", "..."],
  "main_risks": ["...", "..."],
  "recommended_action": "...",
  "summary": "..."
}
"""

    product_owner_prompt_template = """
You are the Product Owner.

You are given the optimized month-level campaign plan:
{optimized_campaign_months}

Your job:
1. Evaluate ROI, expected profit, campaign cost, resource utilization, and return risk.
2. Recommend the best month to run the campaign.
3. Explain trade-offs clearly for business stakeholders.

Return STRICT JSON:
{
  "recommended_month": "...",
  "why_this_month": "...",
  "budget_commentary": "...",
  "capacity_commentary": "...",
  "risk_commentary": "...",
  "final_recommendation": "..."
}
"""

    top_targets_sample = sanitize_for_crewai(top_targets_df.head(5).to_dict(orient="records"))
    optimized_campaign_months = sanitize_for_crewai(optimized_plan_df.to_dict(orient="records"))
    crew_inputs = {
        "top_targets_sample": top_targets_sample,
        "optimized_campaign_months": optimized_campaign_months,
    }

    if not CREWAI_AVAILABLE:
        return None, crew_inputs, "CrewAI is not installed in this environment. Deterministic outputs were still generated."

    solution_analyst_agent = Agent(
        role="Solution Analyst",
        goal="Translate customer-level model outputs into commercial opportunity insights using ROI, uplift, and risk signals.",
        backstory=(
            "You are a commercial analytics specialist who converts predictive outputs into "
            "clear prioritization logic for business teams. You avoid vague narration and focus "
            "on monetization potential, risk, and campaign readiness."
        ),
        verbose=True,
    )
    product_owner_agent = Agent(
        role="Product Owner",
        goal="Recommend the most effective campaign month under budget and resource constraints.",
        backstory=(
            "You are responsible for quarterly campaign planning. You prioritize months that "
            "maximize commercial impact while remaining feasible under cost and operational limits."
        ),
        verbose=True,
    )

    solution_task = Task(
        description=solution_analyst_prompt_template,
        expected_output="Strict JSON describing the priority customer profile and campaign action summary.",
        agent=solution_analyst_agent,
    )
    product_owner_task = Task(
        description=product_owner_prompt_template,
        expected_output="Strict JSON recommending the best campaign month with cost, capacity, and risk commentary.",
        agent=product_owner_agent,
    )

    crew = Crew(
        agents=[solution_analyst_agent, product_owner_agent],
        tasks=[solution_task, product_owner_task],
        process=Process.sequential,
        verbose=True,
    )

    try:
        crew_result = crew.kickoff(inputs=crew_inputs)
        return crew_result, crew_inputs, None
    except Exception as exc:
        return None, crew_inputs, f"Crew execution failed: {exc}"


def run_pipeline(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    raw_df = pd.read_csv(input_path)
    print(f"Loaded data from: {input_path}")
    print(raw_df.shape)

    required_cols = [
        "cust_id",
        "date",
        "online_retail_value",
        "buy_prob",
        "spend_propensity_score",
        "spend_uplift_score",
        "return_prob",
        "recommend_prob",
    ]
    missing = [c for c in required_cols if c not in raw_df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    work_df = raw_df.copy()
    work_df["date"] = pd.to_datetime(work_df["date"], errors="coerce")
    if work_df["date"].isna().any():
        bad_rows = work_df[work_df["date"].isna()].shape[0]
        print(f"Warning: {bad_rows} rows have invalid date and will be dropped.")
        work_df = work_df.dropna(subset=["date"]).copy()

    numeric_cols = [
        "online_retail_value",
        "buy_prob",
        "spend_propensity_score",
        "spend_uplift_score",
        "return_prob",
        "recommend_prob",
    ]
    for col in numeric_cols:
        work_df[col] = pd.to_numeric(work_df[col], errors="coerce")
    work_df = work_df.dropna(subset=numeric_cols).copy()

    for col in ["buy_prob", "spend_propensity_score", "return_prob", "recommend_prob"]:
        work_df[col] = work_df[col].clip(0, 1)
    work_df["spend_uplift_score"] = work_df["spend_uplift_score"].clip(lower=0)

    work_df = add_roi_signals(work_df)
    work_df["campaign_month"] = work_df["date"].dt.to_period("M").astype(str)

    monthly_summary = (
        work_df.groupby("campaign_month", as_index=False)
        .agg(
            customer_count=("cust_id", "count"),
            total_expected_revenue=("expected_total_revenue", "sum"),
            total_expected_profit=("expected_incremental_profit", "sum"),
            total_estimated_campaign_cost=("estimated_campaign_cost", "sum"),
            avg_buy_prob=("buy_prob", "mean"),
            avg_spend_propensity=("spend_propensity_score", "mean"),
            avg_spend_uplift=("spend_uplift_score", "mean"),
            avg_return_prob=("return_prob", "mean"),
            avg_recommend_prob=("recommend_prob", "mean"),
            high_priority_customers=("commercial_priority_band", lambda s: (s == "High").sum()),
            medium_priority_customers=("commercial_priority_band", lambda s: (s == "Medium").sum()),
        )
    )

    monthly_summary["month_level_roi"] = np.where(
        monthly_summary["total_estimated_campaign_cost"] > 0,
        monthly_summary["total_expected_profit"] / monthly_summary["total_estimated_campaign_cost"],
        0,
    )
    monthly_summary["resource_utilization"] = (
        0.50 * (monthly_summary["customer_count"] / monthly_summary["customer_count"].max())
        + 0.30 * (monthly_summary["high_priority_customers"] / monthly_summary["high_priority_customers"].max())
        + 0.20 * (monthly_summary["avg_spend_uplift"] / monthly_summary["avg_spend_uplift"].max())
    ).clip(0, 1)
    monthly_summary["all_in_campaign_cost"] = monthly_summary["total_estimated_campaign_cost"] + fixed_monthly_campaign_cost
    monthly_summary["campaign_objective_score"] = (
        monthly_summary["total_expected_profit"]
        * (1 + 0.20 * monthly_summary["avg_recommend_prob"])
        * (1 - 0.50 * monthly_summary["avg_return_prob"])
    )

    model = LpProblem("ROI_Driven_Campaign_Selection", LpMaximize)
    months = monthly_summary["campaign_month"].tolist()
    x = {m: LpVariable(f"select_{m}", cat=LpBinary) for m in months}
    objective_lookup = dict(zip(monthly_summary["campaign_month"], monthly_summary["campaign_objective_score"]))
    cost_lookup = dict(zip(monthly_summary["campaign_month"], monthly_summary["all_in_campaign_cost"]))
    resource_lookup = dict(zip(monthly_summary["campaign_month"], monthly_summary["resource_utilization"]))

    model += lpSum(objective_lookup[m] * x[m] for m in months)
    model += lpSum(x[m] for m in months) <= max_selected_months
    model += lpSum(cost_lookup[m] * x[m] for m in months) <= quarterly_budget
    model += lpSum(resource_lookup[m] * x[m] for m in months) <= max_resource_utilization
    model.solve()

    selected_months = [m for m in months if x[m].value() == 1]
    optimized_plan_df = (
        monthly_summary[monthly_summary["campaign_month"].isin(selected_months)]
        .sort_values(["campaign_objective_score", "month_level_roi"], ascending=False)
        .reset_index(drop=True)
    )

    top_targets_df = (
        work_df.sort_values(
            ["expected_roi", "expected_incremental_profit", "recommend_prob"],
            ascending=False,
        )[
            [
                "cust_id",
                "campaign_month",
                "online_retail_value",
                "buy_prob",
                "spend_propensity_score",
                "spend_uplift_score",
                "return_prob",
                "recommend_prob",
                "expected_total_revenue",
                "expected_incremental_profit",
                "estimated_campaign_cost",
                "expected_roi",
                "commercial_priority_band",
                "predicted_future_value_summary",
            ]
        ]
        .head(10)
        .reset_index(drop=True)
    )

    crew_result, crew_inputs, crew_error = build_crewai_artifacts(top_targets_df, optimized_plan_df)

    work_df.to_csv(output_dir / "roi_driven_customer_opportunities.csv", index=False)
    monthly_summary.to_csv(output_dir / "roi_driven_monthly_summary.csv", index=False)
    optimized_plan_df.to_csv(output_dir / "optimized_campaign_plan.csv", index=False)
    top_targets_df.to_csv(output_dir / "top_campaign_targets.csv", index=False)
    with open(output_dir / "crew_inputs.json", "w", encoding="utf-8") as f:
        json.dump(crew_inputs, f, indent=2, ensure_ascii=False)
    if crew_result is not None:
        with open(output_dir / "crew_agent_output.txt", "w", encoding="utf-8") as f:
            f.write(str(crew_result))

    print("Export complete:")
    print(f"- {output_dir / 'roi_driven_customer_opportunities.csv'}")
    print(f"- {output_dir / 'roi_driven_monthly_summary.csv'}")
    print(f"- {output_dir / 'optimized_campaign_plan.csv'}")
    print(f"- {output_dir / 'top_campaign_targets.csv'}")
    print(f"- {output_dir / 'crew_inputs.json'}")
    if crew_result is not None:
        print(f"- {output_dir / 'crew_agent_output.txt'}")
    if crew_error:
        print(crew_error)
    if not optimized_plan_df.empty:
        print("\nRecommended month(s):")
        print(optimized_plan_df.to_string(index=False))
    else:
        print("\nNo month was selected under the current budget/resource constraints.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="ROI-driven campaign planner using deterministic optimization plus optional CrewAI.")
    parser.add_argument("--input", default="customer_behavior_with_reviews_scores.csv", help="Input CSV path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for CSV/text exports.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.input), Path(args.output_dir))
