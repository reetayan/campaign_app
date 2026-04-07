from __future__ import annotations

import io
import json
import tempfile
from pathlib import Path

import pandas as pd
import streamlit as st

from campaign_roi_crewai import run_pipeline as run_campaign_pipeline
from model_prediction import run_pipeline as run_model_pipeline

st.set_page_config(
    page_title="Campaign Intelligence Demo",
    page_icon="📈",
    layout="wide",
)


def _save_uploaded_file(uploaded_file, target_path: Path) -> None:
    target_path.write_bytes(uploaded_file.getbuffer())


@st.cache_data(show_spinner=False)
def _read_csv(file_bytes: bytes) -> pd.DataFrame:
    return pd.read_csv(io.BytesIO(file_bytes))


@st.cache_data(show_spinner=False)
def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8") if path.exists() else ""


@st.cache_data(show_spinner=False)
def _read_csv_from_path(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def _read_image_bytes(path: Path) -> bytes | None:
    if path.exists():
        return path.read_bytes()
    return None


st.title("📈 Campaign Intelligence Demo")
st.caption("Streamlit UI for model scoring and ROI-driven campaign planning")

with st.sidebar:
    st.header("How to use")
    st.markdown(
        """
        1. Upload the raw customer behavior CSV in **Model Prediction**.
        2. Run the scoring pipeline to generate prediction outputs.
        3. Upload the scored CSV in **Campaign ROI Planner**.
        4. Run the ROI planner to get the recommended campaign month and top targets.
        """
    )
    st.divider()
    st.markdown("**Expected input for model prediction**")
    st.code(
        "customer_behavior_with_reviews.csv",
        language="text",
    )
    st.markdown("**Expected input for ROI planner**")
    st.code(
        "customer_behavior_with_reviews_scores.csv",
        language="text",
    )

model_tab, campaign_tab = st.tabs(["Model Prediction", "Campaign ROI Planner"])

with model_tab:
    st.subheader("1) Customer Model Prediction")
    st.write(
        "Upload the customer behavior dataset. This runs the prediction pipeline and exports the scored dataset plus charts."
    )

    model_file = st.file_uploader(
        "Upload model input CSV",
        type=["csv"],
        key="model_csv",
    )

    if model_file is not None:
        preview_df = _read_csv(model_file.getvalue())
        st.write("Preview")
        st.dataframe(preview_df.head(10), use_container_width=True)

        with st.expander("Detected columns"):
            st.write(list(preview_df.columns))

        if st.button("Run model prediction", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                input_path = tmpdir_path / model_file.name
                output_dir = tmpdir_path / "outputs"
                _save_uploaded_file(model_file, input_path)

                with st.spinner("Running model prediction pipeline..."):
                    run_model_pipeline(input_path, output_dir)

                st.success("Model prediction pipeline completed.")

                scored_csv = output_dir / "customer_behavior_with_reviews_scores.csv"
                if scored_csv.exists():
                    scored_df = _read_csv_from_path(scored_csv)
                    st.markdown("### Scored Output")
                    st.dataframe(scored_df.head(25), use_container_width=True)
                    st.download_button(
                        "Download scored CSV",
                        data=scored_csv.read_bytes(),
                        file_name="customer_behavior_with_reviews_scores.csv",
                        mime="text/csv",
                    )
                else:
                    st.warning("Scored CSV was not found. Check the pipeline output names in model_prediction.py.")

                metric_images = [
                    ("Model metrics", output_dir / "step1_model_metrics.png"),
                    ("KPI metrics", output_dir / "step2_kpi_metrics.png"),
                    ("High returners", output_dir / "step3a_high_returners.png"),
                    ("Recommenders", output_dir / "step3b_recommenders.png"),
                    ("Campaign pin codes", output_dir / "step3c_campaign_pincodes.png"),
                ]
                for label, image_path in metric_images:
                    image_bytes = _read_image_bytes(image_path)
                    if image_bytes:
                        st.markdown(f"### {label}")
                        st.image(image_bytes, use_container_width=True)
                        st.download_button(
                            f"Download {label.lower()} image",
                            data=image_bytes,
                            file_name=image_path.name,
                            mime="image/png",
                            key=f"dl_{image_path.name}",
                        )

with campaign_tab:
    st.subheader("2) ROI-driven Campaign Planner")
    st.write(
        "Upload the scored dataset and run the ROI planner. This computes opportunity signals, monthly optimization, and optional CrewAI commentary."
    )

    campaign_file = st.file_uploader(
        "Upload scored CSV for campaign planning",
        type=["csv"],
        key="campaign_csv",
    )

    if campaign_file is not None:
        preview_df = _read_csv(campaign_file.getvalue())
        st.write("Preview")
        st.dataframe(preview_df.head(10), use_container_width=True)

        with st.expander("Required columns"):
            st.write([
                "cust_id",
                "date",
                "online_retail_value",
                "buy_prob",
                "spend_propensity_score",
                "spend_uplift_score",
                "return_prob",
                "recommend_prob",
            ])

        if st.button("Run ROI planner", type="primary"):
            with tempfile.TemporaryDirectory() as tmpdir:
                tmpdir_path = Path(tmpdir)
                input_path = tmpdir_path / campaign_file.name
                output_dir = tmpdir_path / "outputs"
                _save_uploaded_file(campaign_file, input_path)

                with st.spinner("Running ROI planner and campaign optimization..."):
                    run_campaign_pipeline(input_path, output_dir)

                st.success("Campaign planning run completed.")

                monthly_path = output_dir / "roi_driven_monthly_summary.csv"
                plan_path = output_dir / "optimized_campaign_plan.csv"
                targets_path = output_dir / "top_campaign_targets.csv"
                opps_path = output_dir / "roi_driven_customer_opportunities.csv"
                crew_inputs_path = output_dir / "crew_inputs.json"
                crew_output_path = output_dir / "crew_agent_output.txt"

                col1, col2, col3 = st.columns(3)
                if plan_path.exists():
                    plan_df = _read_csv_from_path(plan_path)
                    with col1:
                        if not plan_df.empty and "campaign_month" in plan_df.columns:
                            st.metric("Recommended month", str(plan_df.iloc[0]["campaign_month"]))
                        else:
                            st.metric("Recommended month", "No feasible month")
                if monthly_path.exists():
                    monthly_df = _read_csv_from_path(monthly_path)
                    with col2:
                        if not monthly_df.empty and "total_expected_profit" in monthly_df.columns:
                            st.metric("Best expected profit", f"{monthly_df['total_expected_profit'].max():,.2f}")
                if targets_path.exists():
                    targets_df = _read_csv_from_path(targets_path)
                    with col3:
                        st.metric("Top targets surfaced", f"{len(targets_df):,}")

                if plan_path.exists():
                    st.markdown("### Optimized campaign plan")
                    plan_df = _read_csv_from_path(plan_path)
                    st.dataframe(plan_df, use_container_width=True)
                    st.download_button(
                        "Download optimized campaign plan",
                        data=plan_path.read_bytes(),
                        file_name=plan_path.name,
                        mime="text/csv",
                    )

                if monthly_path.exists():
                    st.markdown("### Monthly summary")
                    monthly_df = _read_csv_from_path(monthly_path)
                    st.dataframe(monthly_df, use_container_width=True)
                    st.download_button(
                        "Download monthly summary",
                        data=monthly_path.read_bytes(),
                        file_name=monthly_path.name,
                        mime="text/csv",
                    )

                if targets_path.exists():
                    st.markdown("### Top campaign targets")
                    targets_df = _read_csv_from_path(targets_path)
                    st.dataframe(targets_df, use_container_width=True)
                    st.download_button(
                        "Download top campaign targets",
                        data=targets_path.read_bytes(),
                        file_name=targets_path.name,
                        mime="text/csv",
                    )

                if opps_path.exists():
                    st.markdown("### ROI-driven opportunities")
                    opps_df = _read_csv_from_path(opps_path)
                    st.dataframe(opps_df.head(50), use_container_width=True)
                    st.download_button(
                        "Download opportunity dataset",
                        data=opps_path.read_bytes(),
                        file_name=opps_path.name,
                        mime="text/csv",
                    )

                if crew_inputs_path.exists():
                    st.markdown("### CrewAI inputs")
                    crew_inputs = json.loads(_read_text(crew_inputs_path))
                    st.json(crew_inputs)

                if crew_output_path.exists():
                    st.markdown("### CrewAI output")
                    st.code(_read_text(crew_output_path), language="json")
                else:
                    st.info("CrewAI text output was not generated. This is expected if CrewAI was unavailable or provider configuration was skipped.")

st.divider()
st.caption("Tip: place app.py, model_prediction.py, and campaign_roi_crewai.py in the same folder before running Streamlit.")
