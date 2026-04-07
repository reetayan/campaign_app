"""Converted from ads_021_model_prediction.ipynb.

Usage:
    python model_prediction.py --input customer_behavior_with_reviews.csv --output-dir outputs
"""

from __future__ import annotations

import argparse
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    r2_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

plt.rcParams.update(
    {
        "figure.dpi": 130,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.family": "DejaVu Sans",
        "axes.titlesize": 13,
        "axes.labelsize": 11,
    }
)

POSITIVE = [
    "love", "outstanding", "excellent", "fantastic", "superb", "perfect",
    "incredible", "wonderful", "best", "great", "good", "happy", "satisfied",
    "delighted", "recommend", "amazing", "nice", "solid", "pleasant",
]
NEGATIVE = [
    "terrible", "worst", "horrible", "poor", "disappoint", "regret",
    "useless", "faulty", "defective", "waste", "angry", "horrible",
    "awful", "bad", "below", "unacceptable", "disgusted",
]
FEATURES = [
    "income", "expenses", "quantity", "frequency", "online_retail_value", "cashback",
    "rating", "sentiment_score", "discount_pct", "spend_ratio", "cashback_pct",
    "num_services", "has_emi", "has_subscription", "has_free_returns", "has_loyalty",
    "has_try_buy", "value_per_unit",
    "product_name_enc", "brand_name_enc", "store_name_enc", "card_name_enc", "product_offer_enc",
]


def sentiment_score(text: object) -> float:
    text = str(text).lower()
    pos = sum(1 for w in POSITIVE if w in text)
    neg = sum(1 for w in NEGATIVE if w in text)
    return (pos - neg) / max(pos + neg, 1)


def psi(expected: np.ndarray, actual: np.ndarray, bins: np.ndarray) -> float:
    exp_pct = np.histogram(expected, bins=bins)[0] / len(expected)
    act_pct = np.histogram(actual, bins=bins)[0] / len(actual)
    exp_pct = np.where(exp_pct == 0, 1e-4, exp_pct)
    act_pct = np.where(act_pct == 0, 1e-4, act_pct)
    return np.sum((act_pct - exp_pct) * np.log(act_pct / exp_pct))


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["sentiment_score"] = df["customer_review"].apply(sentiment_score)
    df["discount_pct"] = df["product_offer"].astype(str).str.extract(r"(\d+)").astype(float).fillna(0)
    df["spend_ratio"] = df["expenses"] / (df["income"] + 1)
    df["cashback_pct"] = df["cashback"] / (df["online_retail_value"] + 1)
    df["num_services"] = df["services_used"].apply(lambda x: len(str(x).split("|")))
    df["has_emi"] = df["services_used"].astype(str).str.contains("EMI", na=False).astype(int)
    df["has_subscription"] = df["services_used"].astype(str).str.contains("Subscription", na=False).astype(int)
    df["has_free_returns"] = df["services_used"].astype(str).str.contains("Free Returns", na=False).astype(int)
    df["has_loyalty"] = df["services_used"].astype(str).str.contains("Loyalty", na=False).astype(int)
    df["has_try_buy"] = df["services_used"].astype(str).str.contains("Try & Buy", na=False).astype(int)
    df["value_per_unit"] = df["online_retail_value"] / (df["quantity"] + 1)
    df["pin_prefix"] = (df["pin_code"] // 10000).astype(str)

    for c in ["product_name", "brand_name", "store_name", "card_name", "product_offer"]:
        le = LabelEncoder()
        df[f"{c}_enc"] = le.fit_transform(df[c].astype(str))

    df["buy_label"] = (df["buy"] == "Yes").astype(int)
    return df


def build_step1_plots(
    output_dir: Path,
    auc: float,
    y_test: pd.Series,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    model: GradientBoostingClassifier,
) -> None:
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    ax.plot(fpr, tpr, lw=2.5, label=f"AUC = {auc:.4f}")
    ax.fill_between(fpr, tpr, alpha=0.12)
    ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.2)
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve — Buy Prediction")
    ax.legend(loc="lower right", fontsize=11)

    ax = axes[1]
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", ax=ax,
        xticklabels=["No", "Yes"], yticklabels=["No", "Yes"], linewidths=0.5,
    )
    ax.set_title("Confusion Matrix")
    ax.set_ylabel("Actual")
    ax.set_xlabel("Predicted")

    ax = axes[2]
    fi = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=True).tail(15)
    fi.plot(kind="barh", ax=ax, edgecolor="white", linewidth=0.4)
    ax.set_title("Top 15 Feature Importances")
    ax.set_xlabel("Importance")

    plt.tight_layout()
    plt.savefig(output_dir / "step1_model_metrics.png", bbox_inches="tight")
    plt.close(fig)


def build_step2_plots(
    df: pd.DataFrame,
    output_dir: Path,
    ks_stat: float,
    psi_value: float,
    prob_pos: pd.Series,
    prob_neg: pd.Series,
    train_probs: np.ndarray,
    test_probs: np.ndarray,
    bins: np.ndarray,
) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    ax = axes[0]
    sorted_pos = np.sort(prob_pos)
    sorted_neg = np.sort(prob_neg)
    cdf_pos = np.arange(1, len(sorted_pos) + 1) / len(sorted_pos)
    cdf_neg = np.arange(1, len(sorted_neg) + 1) / len(sorted_neg)
    ax.plot(sorted_pos, cdf_pos, label="Buyers (Yes)", lw=2)
    ax.plot(sorted_neg, cdf_neg, label="Non-buyers (No)", lw=2)
    ax.set_title(f"KS Plot (KS = {ks_stat:.4f})")
    ax.set_xlabel("Predicted Probability")
    ax.set_ylabel("CDF")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    bin_labels = [f"{int(b * 10) * 10}–{int(b * 10) * 10 + 10}%" for b in bins[:-1]]
    train_dist = np.histogram(train_probs, bins=bins)[0] / len(train_probs)
    test_dist = np.histogram(test_probs, bins=bins)[0] / len(test_probs)
    x = np.arange(len(bin_labels))
    w = 0.38
    ax.bar(x - w / 2, train_dist, width=w, label="Train", alpha=0.85)
    ax.bar(x + w / 2, test_dist, width=w, label="Test", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(bin_labels, rotation=40, ha="right", fontsize=8)
    ax.set_title(f"PSI Distribution (PSI = {psi_value:.4f})")
    ax.set_ylabel("Proportion")
    ax.legend()

    ax = axes[2]
    sample = df.sample(min(3000, len(df)), random_state=42)
    sc = ax.scatter(
        sample["spend_propensity_score"],
        sample["spend_uplift_score"],
        c=sample["buy_label"],
        cmap="RdYlGn",
        alpha=0.45,
        s=12,
        edgecolors="none",
    )
    plt.colorbar(sc, ax=ax, label="Buy (1=Yes)")
    ax.set_xlabel("Spend Propensity Score")
    ax.set_ylabel("Spend Uplift Score")
    ax.set_title("Spend Propensity vs Uplift Score")

    plt.tight_layout()
    plt.savefig(output_dir / "step2_kpi_metrics.png", bbox_inches="tight")
    plt.close(fig)


def build_returner_plots(df: pd.DataFrame, returners: pd.DataFrame, threshold: float, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Step 3a — High Return-Risk Customers (vs Full Population)", fontsize=14, y=1.01)

    ax = axes[0, 0]
    ax.hist(df["return_prob"], bins=50, alpha=0.7, label="All", density=True)
    ax.hist(returners["return_prob"], bins=50, alpha=0.75, label="Returners", density=True)
    ax.axvline(threshold, color="black", ls="--", lw=1.4, label=f"Threshold={threshold:.2f}")
    ax.set_title("Return Probability Distribution")
    ax.set_xlabel("Return Prob")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    all_rating = df["rating"].value_counts().sort_index()
    ret_rating = returners["rating"].value_counts().sort_index()
    x = all_rating.index
    w = 0.38
    ax.bar(x - w / 2, all_rating.values / len(df), w, label="All", alpha=0.8)
    ax.bar(x + w / 2, ret_rating.reindex(x, fill_value=0).values / max(len(returners), 1), w, label="Returners", alpha=0.8)
    ax.set_title("Rating — All vs Returners")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.hist(df["sentiment_score"], bins=40, alpha=0.7, label="All", density=True)
    ax.hist(returners["sentiment_score"], bins=40, alpha=0.75, label="Returners", density=True)
    ax.set_title("Sentiment Score — All vs Returners")
    ax.set_xlabel("Sentiment Score")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    ax.hist(df["income"] / 1000, bins=40, alpha=0.7, label="All", density=True)
    ax.hist(returners["income"] / 1000, bins=40, alpha=0.75, label="Returners", density=True)
    ax.set_title("Income Distribution (₹ K)")
    ax.set_xlabel("Income (₹ Thousands)")
    ax.legend(fontsize=8)

    ax = axes[1, 1]
    ax.hist(np.log1p(df["online_retail_value"]), bins=40, alpha=0.7, label="All", density=True)
    ax.hist(np.log1p(returners["online_retail_value"]), bins=40, alpha=0.75, label="Returners", density=True)
    ax.set_title("Online Retail Value (log1p)")
    ax.set_xlabel("log(1 + retail value)")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    vals = [df["cashback_pct"].mean() * 100, returners["cashback_pct"].mean() * 100]
    bars = ax.bar(["All Customers", "Returners"], vals, edgecolor="white", width=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.2, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("Avg Cashback % — All vs Returners")
    ax.set_ylabel("Cashback %")

    plt.tight_layout()
    plt.savefig(output_dir / "step3a_high_returners.png", bbox_inches="tight")
    plt.close(fig)


def build_recommender_plots(df: pd.DataFrame, recommenders: pd.DataFrame, threshold: float, output_dir: Path) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle("Step 3b — High Recommend-to-Neighbour Customers (vs Full Population)", fontsize=14, y=1.01)

    ax = axes[0, 0]
    ax.hist(df["recommend_prob"], bins=50, alpha=0.7, label="All", density=True)
    ax.hist(recommenders["recommend_prob"], bins=50, alpha=0.75, label="Recommenders", density=True)
    ax.axvline(threshold, color="black", ls="--", lw=1.4, label=f"Threshold={threshold:.2f}")
    ax.set_title("Recommend Probability Distribution")
    ax.set_xlabel("Recommend Prob")
    ax.legend(fontsize=8)

    ax = axes[0, 1]
    all_r = df["rating"].value_counts().sort_index()
    rec_r = recommenders["rating"].value_counts().sort_index()
    x = all_r.index
    w = 0.38
    ax.bar(x - w / 2, all_r.values / len(df), w, label="All", alpha=0.8)
    ax.bar(x + w / 2, rec_r.reindex(x, fill_value=0).values / max(len(recommenders), 1), w, label="Recommenders", alpha=0.8)
    ax.set_title("Rating — All vs Recommenders")
    ax.set_xlabel("Rating")
    ax.set_ylabel("Proportion")
    ax.legend(fontsize=8)

    ax = axes[0, 2]
    ax.hist(df["sentiment_score"], bins=40, alpha=0.7, label="All", density=True)
    ax.hist(recommenders["sentiment_score"], bins=40, alpha=0.75, label="Recommenders", density=True)
    ax.set_title("Sentiment Score")
    ax.set_xlabel("Sentiment Score")
    ax.legend(fontsize=8)

    ax = axes[1, 0]
    vals = [df["buy_label"].mean() * 100, recommenders["buy_label"].mean() * 100]
    bars = ax.bar(["All Customers", "Recommenders"], vals, edgecolor="white", width=0.45)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.5, f"{v:.1f}%", ha="center", fontsize=10)
    ax.set_title("Purchase Rate — All vs Recommenders")
    ax.set_ylabel("Buy Rate (%)")
    ax.set_ylim(0, 100)

    ax = axes[1, 1]
    ax.hist(df["spend_propensity_score"], bins=40, alpha=0.7, label="All", density=True)
    ax.hist(recommenders["spend_propensity_score"], bins=40, alpha=0.75, label="Recommenders", density=True)
    ax.set_title("Spend Propensity Score")
    ax.set_xlabel("Propensity")
    ax.legend(fontsize=8)

    ax = axes[1, 2]
    sizes = [len(recommenders), len(df) - len(recommenders)]
    labels = [f"Recommenders\n{len(recommenders):,}", f"Others\n{len(df) - len(recommenders):,}"]
    ax.pie(sizes, labels=labels, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white"})
    ax.set_title("Population Share")

    plt.tight_layout()
    plt.savefig(output_dir / "step3b_recommenders.png", bbox_inches="tight")
    plt.close(fig)


def build_pin_code_plots(top_pins: pd.DataFrame, top5: pd.DataFrame, output_dir: Path) -> None:
    fig = plt.figure(figsize=(20, 14))
    fig.suptitle("Step 3c — Campaign-Ready Pin Code Areas (Next Quarter)", fontsize=15, y=1.01)
    gs = fig.add_gridspec(2, 3, hspace=0.42, wspace=0.35)

    ax1 = fig.add_subplot(gs[0, :2])
    colors = ["#4C72B0" if i < 5 else "#6896c8" if i < 10 else "#9abce0" for i in range(len(top_pins))]
    bars = ax1.bar(top_pins["pin_code"].astype(str), top_pins["campaign_score"], color=colors, edgecolor="white", linewidth=0.5)
    ax1.set_title("Top 20 Pin Codes — Campaign Score (Next Quarter)", fontsize=12)
    ax1.set_xlabel("Pin Code")
    ax1.set_ylabel("Campaign Score (0–1)")
    ax1.set_xticks(range(len(top_pins)))
    ax1.set_xticklabels(top_pins["pin_code"].astype(str), rotation=55, ha="right", fontsize=8)
    ax1.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    for i, (bar, row) in enumerate(zip(bars, top_pins.itertuples())):
        if i < 5:
            ax1.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{row.campaign_score:.2f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")

    ax2 = fig.add_subplot(gs[0, 2])
    pie_data = top5["campaign_score"]
    pie_lbls = top5["pin_code"].astype(str)
    ax2.pie(pie_data, labels=pie_lbls, autopct="%1.1f%%", startangle=90, wedgeprops={"edgecolor": "white", "linewidth": 1.2})
    ax2.set_title("Top 5 Pin Codes — Score Share")

    ax3 = fig.add_subplot(gs[1, :])
    heat_cols = ["avg_propensity", "avg_recommend", "avg_uplift", "buy_rate", "avg_sentiment", "avg_income", "avg_retail_value", "campaign_score"]
    heat_data = top_pins[heat_cols].copy()
    heat_norm = (heat_data - heat_data.min()) / (heat_data.max() - heat_data.min())
    heat_norm.index = top_pins["pin_code"].astype(str).values
    sns.heatmap(
        heat_norm.T,
        ax=ax3,
        cmap="YlOrRd",
        linewidths=0.4,
        linecolor="white",
        annot=heat_data.T.round(2),
        fmt=".2f",
        annot_kws={"size": 7},
        xticklabels=True,
        yticklabels=True,
        cbar_kws={"label": "Normalised Score"},
    )
    ax3.set_title("Heatmap — Top 20 Campaign Pin Codes (multi-dimensional KPIs)", fontsize=11)
    ax3.set_xlabel("Pin Code")
    ax3.set_ylabel("KPI Metric")
    ax3.set_xticklabels(ax3.get_xticklabels(), rotation=45, ha="right", fontsize=8)

    plt.savefig(output_dir / "step3c_campaign_pincodes.png", bbox_inches="tight")
    plt.close(fig)


def run_pipeline(input_path: Path, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(input_path)
    print(f"Loaded data: {input_path}")
    print(f"Shape: {df.shape}")

    df = engineer_features(df)

    X = df[FEATURES]
    y = df["buy_label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

    model = GradientBoostingClassifier(
        n_estimators=400,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.8,
        min_samples_leaf=20,
        random_state=42,
        verbose=0,
    )
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_prob)
    f1 = f1_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    r2 = r2_score(y_test, y_prob)

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=["No (0)", "Yes (1)"]))

    build_step1_plots(output_dir, auc, y_test, y_pred, y_prob, model)

    df["buy_prob"] = model.predict_proba(X)[:, 1]
    df["spend_propensity_score"] = (df["buy_prob"] - df["buy_prob"].min()) / (df["buy_prob"].max() - df["buy_prob"].min())
    raw_uplift = df["spend_propensity_score"] * df["online_retail_value"]
    df["spend_uplift_score"] = (raw_uplift - raw_uplift.min()) / (raw_uplift.max() - raw_uplift.min())

    prob_pos = df.loc[df["buy_label"] == 1, "buy_prob"]
    prob_neg = df.loc[df["buy_label"] == 0, "buy_prob"]
    ks_stat, ks_pval = stats.ks_2samp(prob_pos, prob_neg)

    train_probs = model.predict_proba(X_train)[:, 1]
    test_probs = model.predict_proba(X_test)[:, 1]
    bins = np.linspace(0, 1, 11)
    psi_value = psi(train_probs, test_probs, bins)
    auc_gain = auc - 0.5

    build_step2_plots(df, output_dir, ks_stat, psi_value, prob_pos, prob_neg, train_probs, test_probs, bins)

    df["return_prob"] = (
        (1 - df["sentiment_score"].clip(0, 1)) * 0.4
        + ((5 - df["rating"]) / 4) * 0.4
        + df["cashback_pct"].clip(0, 1) * 0.2
    ).clip(0, 1)
    high_return_thr = df["return_prob"].quantile(0.75)
    returners = df[df["return_prob"] >= high_return_thr]
    build_returner_plots(df, returners, high_return_thr, output_dir)

    df["recommend_prob"] = (
        df["sentiment_score"].clip(0, 1) * 0.35
        + (df["rating"] / 5) * 0.30
        + df["buy_label"] * 0.20
        + df["has_loyalty"] * 0.10
        + df["has_subscription"] * 0.05
    ).clip(0, 1)
    high_rec_thr = df["recommend_prob"].quantile(0.75)
    recommenders = df[df["recommend_prob"] >= high_rec_thr]
    build_recommender_plots(df, recommenders, high_rec_thr, output_dir)

    pin_stats = df.groupby("pin_code").agg(
        customer_count=("cust_id", "count"),
        avg_propensity=("spend_propensity_score", "mean"),
        avg_uplift=("spend_uplift_score", "mean"),
        avg_return_risk=("return_prob", "mean"),
        avg_recommend=("recommend_prob", "mean"),
        buy_rate=("buy_label", "mean"),
        avg_income=("income", "mean"),
        avg_retail_value=("online_retail_value", "mean"),
        avg_sentiment=("sentiment_score", "mean"),
    ).reset_index()

    pin_stats["campaign_score"] = (
        pin_stats["avg_propensity"] * 0.30
        + pin_stats["avg_recommend"] * 0.25
        + pin_stats["avg_uplift"] * 0.25
        + (1 - pin_stats["avg_return_risk"]) * 0.20
    )
    pin_stats["campaign_score"] = (pin_stats["campaign_score"] - pin_stats["campaign_score"].min()) / (
        pin_stats["campaign_score"].max() - pin_stats["campaign_score"].min()
    )
    pin_stats = pin_stats.sort_values("campaign_score", ascending=False)
    top_pins = pin_stats.head(20)
    top5 = pin_stats.head(5)
    build_pin_code_plots(top_pins, top5, output_dir)

    df.to_csv(output_dir / "customer_behavior_with_predictions.csv", index=False)
    pin_stats.to_csv(output_dir / "campaign_ready_pincodes.csv", index=False)

    print("=" * 55)
    print("  ADS-021 Model Prediction — Analysis Complete")
    print("=" * 55)
    print(f"  Total Customers Analysed : {len(df):,}")
    print(f"  Model AUC-ROC            : {auc:.4f}")
    print(f"  Model R²                 : {r2:.4f}")
    print(f"  KS Statistic             : {ks_stat:.4f}")
    print(f"  KS p-value               : {ks_pval:.4f}")
    print(f"  PSI (stability)          : {psi_value:.4f}")
    print(f"  AUC Gain                 : {auc_gain:.4f}")
    print(f"  Accuracy                 : {acc:.4f}")
    print(f"  F1                       : {f1:.4f}")
    print(f"  Precision                : {prec:.4f}")
    print(f"  Recall                   : {rec:.4f}")
    print(f"  High Return-Risk Cust.   : {len(returners):,} ({len(returners) / len(df) * 100:.1f}%)")
    print(f"  High Recommender Cust.   : {len(recommenders):,} ({len(recommenders) / len(df) * 100:.1f}%)")
    print(f"  Top Campaign Pin Code    : {pin_stats.iloc[0]['pin_code']}")
    print(f"  Outputs directory        : {output_dir}")
    print("=" * 55)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Customer predictive analytics model pipeline.")
    parser.add_argument("--input", default="customer_behavior_with_reviews.csv", help="Input CSV path.")
    parser.add_argument("--output-dir", default="outputs", help="Directory for charts and CSV exports.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run_pipeline(Path(args.input), Path(args.output_dir))
