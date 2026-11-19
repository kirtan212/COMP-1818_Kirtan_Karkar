"""
Publication-Quality Visualization Module
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import os
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
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
    Plots horizontal bar chart comparing Nested 5x3 CV Macro F1 (with error bars) and Test Macro F1.
    """
    os.makedirs(output_dir, exist_ok=True)
    df_sorted = results_df.sort_values(by="CV Macro F1 Mean", ascending=True).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.6, 5.0))
    y_pos = np.arange(len(df_sorted))
    bar_height = 0.36

    ax.barh(
        y_pos - bar_height / 2,
        df_sorted["CV Macro F1 Mean"],
        height=bar_height,
        xerr=df_sorted["CV Macro F1 Std"],
        capsize=3.5,
        color="#2b5c8f",
        edgecolor="black",
        linewidth=0.8,
        label="Nested $5\\times 3$ CV Macro F1 (Mean $\\pm \\sigma$)"
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
    ax.set_title("Benchmark Comparison: Baselines, Symbolic Rules, and ML Models")
    ax.set_xlim(0.0, 1.0)
    ax.legend(loc="lower right", frameon=True)

    for i in y_pos:
        cv_val = df_sorted.loc[i, "CV Macro F1 Mean"]
        cv_std = df_sorted.loc[i, "CV Macro F1 Std"]
        test_val = df_sorted.loc[i, "Test Macro F1"]
        ax.text(cv_val + cv_std + 0.015, i - bar_height / 2, f"{cv_val:.3f}", va="center", fontsize=8, color="#1a3b5c")
        ax.text(test_val + 0.015, i + bar_height / 2, f"{test_val:.3f}", va="center", fontsize=8, color="#8b2f22")

    plt.tight_layout()
    out_path = os.path.join(output_dir, "model_comparison.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_feature_importance(shap_df, perm_df, top_n=8, output_dir="figures"):
    """
    Generates a side-by-side two-panel explainability figure for the full
    RF + XGBoost Soft-Voting Ensemble:
      (a) Probability-space Ensemble TreeSHAP (Mean |SHAP| and % share)
      (b) Full-pipeline Permutation Importance (Macro F1 drop +- std)
    """
    os.makedirs(output_dir, exist_ok=True)
    top_shap = shap_df.head(top_n).sort_values(by="Mean_Abs_SHAP", ascending=True)
    top_perm = perm_df.head(top_n).sort_values(by="Perm_F1_Drop_Mean", ascending=True)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10.6, 4.3))

    # Panel (a): Probability-space Ensemble SHAP
    y1 = np.arange(len(top_shap))
    bars1 = ax1.barh(y1, top_shap["Mean_Abs_SHAP"], color="#2b9348", edgecolor="#1b4332", linewidth=0.8)
    ax1.set_yticks(y1)
    ax1.set_yticklabels(top_shap["Feature"], fontsize=9)
    ax1.set_xlabel("Mean $|\\phi_j|$ (Approval Probability Shift)")
    ax1.set_title("(a) Ensemble Probability-Space SHAP")
    max_s = float(top_shap["Mean_Abs_SHAP"].max()) if len(top_shap) else 0.2
    ax1.set_xlim(0.0, max_s * 1.48)
    for bar, pct in zip(bars1, top_shap["SHAP_Share_Pct"]):
        w = bar.get_width()
        ax1.text(w + max_s * 0.02, bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f} ({pct:.1f}%)", va="center", fontsize=8)

    # Panel (b): Full-pipeline Permutation Importance
    y2 = np.arange(len(top_perm))
    bars2 = ax2.barh(
        y2,
        top_perm["Perm_F1_Drop_Mean"],
        xerr=top_perm["Perm_F1_Drop_Std"],
        capsize=3,
        color="#2b5c8f",
        edgecolor="#0f172a",
        linewidth=0.8
    )
    ax2.axvline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    ax2.set_yticks(y2)
    ax2.set_yticklabels(top_perm["Feature"], fontsize=9)
    ax2.set_xlabel("Mean Macro F1 Drop (30 Repeats)")
    ax2.set_title("(b) Full-Ensemble Permutation Importance")
    max_p = float((top_perm["Perm_F1_Drop_Mean"] + top_perm["Perm_F1_Drop_Std"]).max()) if len(top_perm) else 0.2
    min_p = min(0.0, float((top_perm["Perm_F1_Drop_Mean"] - top_perm["Perm_F1_Drop_Std"]).min()))
    ax2.set_xlim(min_p - 0.02, max_p * 1.28)
    for bar, std in zip(bars2, top_perm["Perm_F1_Drop_Std"]):
        w = bar.get_width()
        x_txt = max(w, 0.0) + std + 0.008
        ax2.text(x_txt, bar.get_y() + bar.get_height() / 2,
                 f"{w:.3f}", va="center", fontsize=8)

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

    colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#17becf"]

    for idx, (name, prob) in enumerate(probabilities_dict.items()):
        if name in ["Majority Baseline", "Credit History One-Rule", "Random Baseline"]:
            continue
        c = colors[idx % len(colors)]
        fpr, tpr, _ = roc_curve(y_test, prob)
        roc_val = auc(fpr, tpr)
        ax1.plot(fpr, tpr, color=c, lw=1.8, label=f"{name} (AUC = {roc_val:.3f})")

        prec, rec, _ = precision_recall_curve(y_test, prob)
        pr_val = auc(rec, prec)
        ax2.plot(rec, prec, color=c, lw=1.8, label=f"{name} (PR-AUC = {pr_val:.3f})")

    ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.6, label="Chance Baseline (AUC = 0.500)")
    ax1.set_xlabel("False Positive Rate (1 - Specificity)")
    ax1.set_ylabel("True Positive Rate (Sensitivity)")
    ax1.set_title("(a) Receiver Operating Characteristic (ROC)")
    ax1.legend(loc="lower right", fontsize=8)
    ax1.set_xlim([0.0, 1.0])
    ax1.set_ylim([0.0, 1.05])

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
        "Credit History One-Rule",
        "Symbolic Expert Rules",
        "Logistic Regression",
        "RF + XGBoost Soft Voting"
    ]

    fig, axes = plt.subplots(2, 2, figsize=(8.2, 6.8))
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
            ax.set_ylabel("True Label")
            ax.set_xticklabels(["Rejected (0)", "Approved (1)"])
            ax.set_yticklabels(["Rejected (0)", "Approved (1)"])

    plt.suptitle("Normalized Confusion Matrices on Held-Out Test Set ($N=123$)", y=0.99, fontsize=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "confusion_matrices.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def plot_cost_threshold_analysis(
    thresholds, oof_costs_per_100, test_costs_per_100, opt_threshold, output_dir="figures"
):
    """
    Plots leak-free asymmetric lending cost per 100 applications:
      - Solid blue curve: Training OOF cost used strictly to select tau*
      - Dashed red curve: Held-out test cost evaluated at the frozen tau* vs tau=0.50
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7.6, 4.4))

    opt_idx = int(np.argmin(np.abs(thresholds - opt_threshold)))
    def_idx = int(np.argmin(np.abs(thresholds - 0.50)))

    ax.plot(
        thresholds, oof_costs_per_100, color="#2b5c8f", lw=2.2,
        label="Training OOF Loss ($N=491$, selects $\\tau^*$)"
    )
    ax.plot(
        thresholds, test_costs_per_100, color="#d90429", lw=1.8, linestyle="--", alpha=0.85,
        label="Held-Out Test Loss ($N=123$, evaluation only)"
    )
    ax.axvline(0.50, color="#6c757d", linestyle=":", lw=1.5, label="Default Cutoff ($\\tau=0.50$)")
    ax.axvline(
        opt_threshold, color="#2b9348", linestyle="-.", lw=1.8,
        label=f"OOF-Selected Cutoff ($\\tau^*={opt_threshold:.2f}$)"
    )

    ax.scatter([0.50], [test_costs_per_100[def_idx]], color="#6c757d", s=55, zorder=5)
    ax.scatter([opt_threshold], [oof_costs_per_100[opt_idx]], color="#2b5c8f", s=60, zorder=5)
    ax.scatter([opt_threshold], [test_costs_per_100[opt_idx]], color="#2b9348", s=65, zorder=6)

    ax.set_xlabel("Classification Decision Threshold ($\\tau$)")
    ax.set_ylabel("Asymmetric Risk Penalty per 100 Apps ($5\\cdot\\mathrm{FP} + 1\\cdot\\mathrm{FN}$)")
    ax.set_title("Leak-Free Asymmetric Cost Threshold Tuning (LO2)")
    ax.legend(loc="upper center", frameon=True, fontsize=8.5)

    plt.tight_layout()
    out_path = os.path.join(output_dir, "calibration_cost_tradeoff.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path


def generate_architecture_diagram(output_dir="figures"):
    """
    Generates figure01.png - the methodological architecture flowchart.
    """
    os.makedirs(output_dir, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4.0))
    ax.axis("off")

    boxes = [
        {"x": 0.04, "y": 0.30, "w": 0.19, "h": 0.56, "title": "Data Ingestion\n& Preprocessing",
         "desc": "• 614 Applications\n• 80/20 Stratified Split\n• Median/Mode Impute\n• OHE + Z-Score Scaler", "color": "#e0f2fe"},
        {"x": 0.28, "y": 0.30, "w": 0.19, "h": 0.56, "title": "Baseline & Symbolic\nFormalisms (LO3)",
         "desc": "• Majority Baseline\n• Stratified Random\n• 1-Rule Credit History\n• LTI Expert Rules", "color": "#fef3c7"},
        {"x": 0.52, "y": 0.30, "w": 0.19, "h": 0.56, "title": "Nested 5x3 CV\nModel Tuning",
         "desc": "• Logistic Regression\n• Decision Tree / SVM\n• Random Forest\n• XGBoost (Macro F1)", "color": "#e0e7ff"},
        {"x": 0.76, "y": 0.30, "w": 0.20, "h": 0.56, "title": "Soft Voting, XAI\n& OOF Risk (LO2)",
         "desc": "• RF + XGBoost Voting\n• Ensemble SHAP + Perm\n• OOF Cost Cutoff $\\tau^*$\n• Saved Joblib Models", "color": "#dcfce7"}
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
            b["x"] + 0.015, b["y"] + 0.17, b["desc"],
            ha="left", va="center", transform=ax.transAxes, fontsize=8, color="#334155"
        )

    arrow_props = dict(facecolor="#475569", edgecolor="#334155", width=1.8, headwidth=6, headlength=6)
    ax.annotate("", xy=(0.28, 0.58), xytext=(0.23, 0.58), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.52, 0.58), xytext=(0.47, 0.58), xycoords="axes fraction", arrowprops=arrow_props)
    ax.annotate("", xy=(0.76, 0.58), xytext=(0.71, 0.58), xycoords="axes fraction", arrowprops=arrow_props)

    ax.set_title("End-to-End Methodological Architecture of the Explainable Loan Approval System", fontsize=11, fontweight="bold", pad=12)
    plt.tight_layout()
    out_path = os.path.join(output_dir, "figure01.png")
    plt.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close()
    return out_path
