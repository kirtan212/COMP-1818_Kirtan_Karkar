"""
Publication-Quality Visualization Module
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, precision_recall_curve, auc, confusion_matrix

# Set global academic plotting aesthetic
sns.set_theme(style="whitegrid", font="DejaVu Sans")
plt.rcParams.update({
    "font.size": 10,
    "axes.labelsize": 11,
    "axes.titlesize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 9,
    "figure.titlesize": 13,
    "figure.dpi": 300
})


def plot_model_comparison(results_df, output_dir="figures"):
    """
    Plots horizontal bar chart comparing Nested CV Macro F1 (with error bars) and Test Macro F1.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_sorted = results_df.sort_values(by="CV Macro F1 Mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9, 5.5))
    y_pos = np.arange(len(df_sorted))
    bar_height = 0.36

    ax.barh(
        y_pos - bar_height / 2,
        df_sorted["CV Macro F1 Mean"],
        height=bar_height,
        xerr=df_sorted["CV Macro F1 Std"],
        capsize=4,
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        label="Nested 5-Fold CV Macro F1 (Mean $\\pm \\sigma$)"
    )

    ax.barh(
        y_pos + bar_height / 2,
        df_sorted["Test Macro F1"],
        height=bar_height,
        color="#e26d5c",
        edgecolor="black",
        linewidth=0.8,
        label="Independent Test Macro F1 ($N=123$)"
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(df_sorted["Model"], fontweight="medium")
    ax.set_xlabel("Macro-Averaged F1 Score")
    ax.set_title("Benchmark Comparison Across Trivial Baselines and Machine Learning Models")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=True)

    for i in y_pos:
        cv_val = df_sorted.loc[i, "CV Macro F1 Mean"]
        test_val = df_sorted.loc[i, "Test Macro F1"]
        ax.text(cv_val + 0.02, i - bar_height / 2, f"{cv_val:.3f}", va="center", fontsize=8, color="#1a3b5c")
        ax.text(test_val + 0.02, i + bar_height / 2, f"{test_val:.3f}", va="center", fontsize=8, color="#8b2f22")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_feature_importance(importance_df, top_n=10, output_dir="figures"):
    """
    Plots horizontal bar chart of top ranking feature importances.
    """
    os.makedirs(output_dir, exist_ok=True)
    top_df = importance_df.head(top_n).sort_values(by="Importance", ascending=True)

    fig, ax = plt.subplots(figsize=(8, 4.8))
    y_pos = np.arange(len(top_df))

    bars = ax.barh(y_pos, top_df["Importance"], color="#38b000", edgecolor="#1b4332", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_df["Feature"])
    ax.set_xlabel("Mean Absolute Feature Attribution / Importance")
    ax.set_title(f"Top {top_n} Predictor Attribution in Ensemble Loan Decision Modeling")

    for bar in bars:
        w = bar.get_width()
        ax.text(w + 0.005, bar.get_y() + bar.get_height() / 2, f"{w:.3f}", va="center", fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "feature_importance.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_roc_pr_curves(probabilities_dict, y_test, output_dir="figures"):
    """
    Generates side-by-side ROC and Precision-Recall curves.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.8))

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

    for idx, (name, prob) in enumerate(probabilities_dict.items()):
        if name in ["Majority Baseline", "Credit History One-Rule", "Random Baseline"]:
            continue
        c = colors[idx % len(colors)]
        # ROC
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=c, lw=1.8, label=f"{name} (AUC = {roc_val:.3f})")

        # PR
        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_val = auc(rec, prec)
        ax2.plot(rec, prec, color=c, lw=1.8, label=f"{name} (PR-AUC = {pr_val:.3f})")

    # ROC diagonal
    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance Baseline (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("(a) Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

    # PR horizontal line (prior)
    prior = np.mean(y_test)
    ax2.plot([0, 1], [prior, prior], "k--", lw=1, alpha=0.6, label=f"No-Skill Prior ({prior:.2f})")
    ax2.set_xlabel("Recall")
    ax2.set_ylabel("Precision")
    ax2.set_title("(b) Precision-Recall (PR) Curve")
    ax2.legend(loc="lower left", fontsize=8)
    ax2.set_xlim([0.0, 1.0])
    ax2.set_ylim([0.0, 1.05])

    plt.tight_layout()
    out_path = os.path.join(output_dir, "roc_pr_curves.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_confusion_matrices(predictions_dict, y_test, output_dir="figures"):
    """
    Plots a 2x2 comparison grid of normalized confusion matrices.
    """
    os.makedirs(output_dir, exist_ok=True)
    models_to_plot = [
        "Majority Baseline",
        "Credit History One-Rule",
        "Logistic Regression",
        "RF + XGBoost Soft Voting"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
    axes = axes.flatten()

    for idx, name in enumerate(models_to_plot):
        ax = axes[idx]
        if name in predictions_dict:
            preds = predictions_dict[name]
            cm = confusion_matrix(y_test, preds)
            cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

            sns.heatmap(
                cm_norm, annot=False, cmap="Blues", cbar=False, ax=ax, vmin=0, vmax=1
            )
            # Annotate with both percentage and count
            for r in range(2):
                for c in range(2):
                    count = cm[r, c]
                    pct = cm_norm[r, c] * 100
                    color = "white" if pct > 50 else "black"
                    ax.text(
                        c + 0.5, r + 0.5, f"{pct:.1f}%\n(n={count})",
                        ha="center", va="center", color=color, fontweight="semibold", fontsize=9.5
                    )

            ax.set_title(f"{name}", fontsize=11)
            ax.set_xlabel("Predicted Label")
            ax.set_ylabel("True Ground-Truth Label")
            ax.set_xticklabels(["Rejected (0)", "Approved (1)"])
            ax.set_yticklabels(["Rejected (0)", "Approved (1)"])

    plt.suptitle("Normalized Confusion Matrices on Held-Out Test Set ($N=123$)", y=0.99, fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_cost_threshold_analysis(thresholds, total_costs, optimal_threshold, output_dir="figures"):
    """
    Plots asymmetric portfolio risk cost vs classification decision threshold.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.5, 4.5))

    ax.plot(thresholds, total_costs, color="#d90429", lw=2.2, label="Total Risk Cost ($5\\cdot FP + 1\\cdot FN$)")
    ax.axvline(0.50, color="#6c757d", linestyle="--", lw=1.5, label="Default Cutoff ($\\tau=0.50$)")
    ax.axvline(optimal_threshold, color="#2b9348", linestyle="-.", lw=1.8, label=f"Risk-Optimal Cutoff ($\\tau^*={optimal_threshold:.2f}$)")

    ax.scatter([optimal_threshold], [np.min(total_costs)], color="#2b9348", s=60, zorder=5)
    ax.set_xlabel("Classification Decision Threshold ($\\tau$)")
    ax.set_ylabel("Asymmetric Monetary Penalty / Portfolio Loss")
    ax.set_title("Credit Decision Threshold Optimization Under Asymmetric Default Costs (LO2)")
    ax.legend(loc="upper center", frameon=True)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "calibration_cost_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


from matplotlib.patches import FancyBboxPatch

def generate_architecture_diagram(output_dir="figures"):
    """
    Generates figure01.png - the methodological architecture flowchart.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.2))
    ax.axis("off")

    # Drawing boxes representing the end-to-end framework
    boxes = [
        {"x": 0.05, "y": 0.35, "w": 0.18, "h": 0.52, "title": "Data Ingestion\n& Preprocessing",
         "desc": "• 614 Applications\n• Median Imputation\n• One-Hot Encoding\n• Z-Score Scaler", "color": "#e0f2fe"},
        {"x": 0.29, "y": 0.35, "w": 0.18, "h": 0.52, "title": "Baseline & Symbolic\nFormalisms (LO3)",
         "desc": "• Majority Class\n• Random Guess\n• 1-Rule Credit\n• Expert Rules", "color": "#fef3c7"},
        {"x": 0.53, "y": 0.35, "w": 0.18, "h": 0.52, "title": "Nested 5x5 CV\nModel Tuning",
         "desc": "• Logistic Regression\n• Decision Trees\n• Random Forest\n• Support Vector", "color": "#e0e7ff"},
        {"x": 0.77, "y": 0.35, "w": 0.19, "h": 0.52, "title": "Explainable Soft\nVoting & Risk (LO2)",
         "desc": "• RF + XGBoost Voting\n• SHAP Attribution\n• Asymmetric Cost $\\tau^*$\n• Fair Lending Audit", "color": "#dcfce7"}
    ]

    for b in boxes:
        rect = FancyBboxPatch(
            (b["x"], b["y"]), b["w"], b["h"],
            boxstyle="round,pad=0.02,rounding_size=0.03",
            facecolor=b["color"], edgecolor="#1e293b", linewidth=1.2, transform=ax.transAxes
        )
        ax.add_patch(rect)
        ax.text(
            b["x"] + b["w"] / 2, b["y"] + b["h"] - 0.10, b["title"],
            ha="center", va="center", transform=ax.transAxes, fontsize=9.5, fontweight="bold", color="#0f172a"
        )
        ax.text(
            b["x"] + 0.015, b["y"] + 0.16, b["desc"],
            ha="left", va="center", transform=ax.transAxes, fontsize=8, color="#334155"
        )

    # Arrows between stages
    arrow_props = dict(facecolor="#475569", edgecolor="#334155", width=1.8, headwidth=6, headlength=6)
    ax.annotate("", xy=(0.29, 0.61), xytext=(0.23, 0.61), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.53, 0.61), xytext=(0.47, 0.61), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.77, 0.61), xytext=(0.71, 0.61), xycoords="axes fraction", arrowprops=arrow_props)

    ax.set_title("Figure 1: End-to-End Methodological Architecture of the Explainable Loan Approval System", fontsize=11, fontweight="bold", pad=15)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure01.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path
