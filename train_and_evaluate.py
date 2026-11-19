"""
Master End-to-End Training, Cross-Validation, and Evaluation Pipeline
COMP1818 Artificial Intelligence Applications - University of Greenwich
Author: Kirtan Karkar
"""

import os
import re
import json
import shutil
import time
import joblib
import numpy as np

from src.data_loader import get_data_splits, build_preprocessor
from src.models import build_model_pipelines, get_hyperparameter_grids
from src.evaluate import (
    run_nested_cross_validation,
    evaluate_on_test_set,
    merge_and_save_results
)
from src.explainability import (
    compute_feature_importance,
    analyze_asymmetric_lending_costs_leak_free
)
from src.plotting import (
    plot_model_comparison,
    plot_feature_importance,
    plot_roc_pr_curves,
    plot_confusion_matrices,
    plot_cost_threshold_analysis,
    generate_architecture_diagram
)


def _slugify(name):
    return re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")


def save_trained_models(fitted_models, cv_summary, opt_threshold, models_dir):
    """Persists all trained model pipelines to disk with joblib plus a JSON manifest."""
    os.makedirs(models_dir, exist_ok=True)
    params_lookup = dict(zip(cv_summary["Model"], cv_summary["Best Params"]))
    manifest = {
        "oof_selected_threshold_rf_xgb_soft_voting": round(float(opt_threshold), 2),
        "cost_ratio_fp_to_fn": "5:1",
        "models": {}
    }

    for name, model in fitted_models.items():
        fname = f"{_slugify(name)}.joblib"
        fpath = os.path.join(models_dir, fname)
        joblib.dump(model, fpath, compress=3)
        manifest["models"][name] = {
            "file": fname,
            "best_params": params_lookup.get(name, {})
        }

    with open(os.path.join(models_dir, "model_manifest.json"), "w") as f:
        json.dump(manifest, f, indent=4)


def main():
    start_time = time.time()
    print("=" * 80)
    print(" COMP1818 AI APPLICATIONS - LOAN APPROVAL PREDICTION BENCHMARK")
    print(" Explainable Ensemble Learning Framework & Symbolic Baseline Comparison")
    print("=" * 80)

    # 1. Paths and Setup
    base_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(base_dir, "data", "train.csv")
    fig_dir = os.path.join(base_dir, "figures")
    out_dir = os.path.join(base_dir, "outputs")
    models_dir = os.path.join(base_dir, "models")
    report_dir = os.path.join(base_dir, "report")

    for d in (fig_dir, out_dir, models_dir, report_dir):
        os.makedirs(d, exist_ok=True)

    # 2. Data Loading & Preprocessing
    print("\n[Phase 1/5] Ingesting and Partitioning Dataset...")
    X_train, X_test, y_train, y_test, num_cols, cat_cols = get_data_splits(
        filepath=data_path, test_size=0.20, random_state=42
    )
    print(f"  * Total Applications: {len(X_train) + len(X_test)}")
    print(f"  * Training Records (80%): {len(X_train)} ({np.sum(y_train==1)} Approved, {np.sum(y_train==0)} Rejected)")
    print(f"  * Testing Records (20%): {len(X_test)} ({np.sum(y_test==1)} Approved, {np.sum(y_test==0)} Rejected)")
    print(f"  * Numerical Predictors: {num_cols}")
    print(f"  * Categorical Predictors: {cat_cols}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 3. Model Pipeline Construction
    print("\n[Phase 2/5] Building Baseline, Symbolic (LO3), and ML Model Pipelines...")
    models = build_model_pipelines(preprocessor, random_state=42)
    param_grids = get_hyperparameter_grids()
    print(f"  * Initialized {len(models)} model architectures across 4 paradigms:")
    print("    1. Trivial Statistical Baselines: Majority Class, Stratified Random")
    print("    2. Symbolic Rule Baselines: Credit History 1-Rule, LTI Expert Rules Engine (LO3)")
    print("    3. Supervised Machine Learning: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost")
    print("    4. Proposed Extension: Soft-Voting Ensemble (Random Forest + XGBoost)")

    # 4. Nested Cross-Validation Execution (5 outer x 3 inner)
    print("\n[Phase 3/5] Executing Nested Stratified 5x3 Cross-Validation...")
    cv_summary, fold_scores, fitted_models, oof_probs_dict = run_nested_cross_validation(
        models=models,
        param_grids=param_grids,
        X_train=X_train,
        y_train=y_train,
        outer_folds=5,
        inner_folds=3,
        random_state=42
    )

    # Log Symbolic Expert Rules firing diagnostics
    sym_expert = fitted_models["Symbolic Expert Rules"]
    print("\n  * Symbolic Expert Rules Firing Counts (Train, N=491):", sym_expert.count_rule_firings(X_train))
    print("  * Symbolic Expert Rules Firing Counts (Test,  N=123):", sym_expert.count_rule_firings(X_test))

    # 5. Independent Test Set Evaluation
    print("\n[Phase 4/5] Evaluating Fitted Models on Independent Test Set (N=123)...")
    test_summary, preds_dict, probs_dict = evaluate_on_test_set(
        fitted_models=fitted_models,
        X_test=X_test,
        y_test=y_test
    )

    results_df = merge_and_save_results(
        cv_df=cv_summary,
        test_df=test_summary,
        fold_df=fold_scores,
        output_dir=out_dir
    )

    # 6. Explainability (Ensemble SHAP + Permutation) & Leak-Free Asymmetric Cost Tuning
    print("\n[Phase 5/5] Computing Ensemble Explainability & Leak-Free Risk Cost Optimisation...")
    ensemble_model = fitted_models["RF + XGBoost Soft Voting"]
    shap_df, perm_df, _ = compute_feature_importance(
        model=ensemble_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=out_dir
    )
    print("  * Top 3 Ensemble Probability-Space SHAP Features:")
    for _, r in shap_df.head(3).iterrows():
        print(f"    - {r['Feature']}: Mean |SHAP| = {r['Mean_Abs_SHAP']:.4f} ({r['SHAP_Share_Pct']:.1f}%)")
    print("  * Top 3 Full-Ensemble Permutation Features (Macro F1 drop):")
    for _, r in perm_df.head(3).iterrows():
        print(f"    - {r['Feature']}: {r['Perm_F1_Drop_Mean']:.4f} +- {r['Perm_F1_Drop_Std']:.4f}")

    oof_ens_probs = oof_probs_dict["RF + XGBoost Soft Voting"]
    test_ens_probs = probs_dict["RF + XGBoost Soft Voting"]
    thresholds, oof_per_100, test_per_100, opt_threshold, cost_summary = (
        analyze_asymmetric_lending_costs_leak_free(
            y_train=y_train.to_numpy(),
            oof_prob_train=oof_ens_probs,
            y_test=y_test.to_numpy(),
            prob_test=test_ens_probs,
            cost_fp=5.0,
            cost_fn=1.0,
            output_dir=out_dir
        )
    )
    oof_s = cost_summary["oof_train"]
    tst_s = cost_summary["heldout_test"]
    print("  * Leak-Free Asymmetric Lending Risk Analysis (5:1 FP:FN Cost Ratio):")
    print(f"    - Selected Cutoff on Training OOF CV: tau* = {opt_threshold:.2f}")
    print(f"    - Training OOF Loss: {oof_s['default_tau_050_cost']:.1f} (tau=0.50) -> {oof_s['opt_tau_cost']:.1f} (tau*={opt_threshold:.2f}), reduction = {oof_s['cost_reduction_pct']:.1f}%")
    print(f"    - Held-Out Test Loss: {tst_s['default_tau_050_cost']:.1f} (FP={tst_s['default_tau_050_fp']}, FN={tst_s['default_tau_050_fn']}) -> {tst_s['opt_tau_cost']:.1f} (FP={tst_s['opt_tau_fp']}, FN={tst_s['opt_tau_fn']}), out-of-sample reduction = {tst_s['cost_reduction_pct']:.1f}%")

    # Save trained models and manifest
    save_trained_models(fitted_models, cv_summary, opt_threshold, models_dir)
    print(f"  * Saved {len(fitted_models)} trained model pipelines to {models_dir}/")

    # 7. Generating Publication-Quality Figures
    print("\nGenerating Figures...")
    p1 = plot_model_comparison(results_df, output_dir=fig_dir)
    p2 = plot_feature_importance(shap_df, perm_df, top_n=8, output_dir=fig_dir)
    p3 = plot_roc_pr_curves(probs_dict, y_test, output_dir=fig_dir)
    p4 = plot_confusion_matrices(preds_dict, y_test, output_dir=fig_dir)
    p5 = plot_cost_threshold_analysis(
        thresholds, oof_per_100, test_per_100, opt_threshold, output_dir=fig_dir
    )
    p6 = generate_architecture_diagram(output_dir=fig_dir)

    # Copy the figures used in the 4-page LaTeX report into report/
    for fig_file in [p1, p2, p4, p5]:
        shutil.copy(fig_file, report_dir)
    # Remove unused duplicate figures in report/ to keep report/ lean
    for unused in ("figure01.png", "roc_pr_curves.png"):
        unused_path = os.path.join(report_dir, unused)
        if os.path.exists(unused_path):
            os.remove(unused_path)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" EXECUTION COMPLETE (Runtime: {elapsed:.2f} seconds)")
    print("=" * 80)
    print("\nQUANTITATIVE BENCHMARK SUMMARY:")
    display_cols = ["Model", "Test Accuracy", "Test Macro F1", "Test ROC AUC", "CV Macro F1 Mean", "CV Macro F1 Std"]
    print(results_df[display_cols].to_string(index=False))

    print("\nArtifacts and deliverables generated:")
    print(f"  * Trained Models:    {models_dir}/")
    print(f"  * Tabular Results:   {os.path.join(out_dir, 'evaluation_results.csv')}")
    print(f"  * Explainability:    {os.path.join(out_dir, 'feature_importance.csv')}, permutation_importance.csv")
    print(f"  * Cost Analysis:     {os.path.join(out_dir, 'cost_threshold_analysis.json')}")
    print(f"  * LaTeX Table Code:  {os.path.join(out_dir, 'latex_results_table.tex')}")
    print(f"  * All Figures:       {fig_dir}/")
    print(f"  * Report Assets:     {report_dir}/")


if __name__ == "__main__":
    main()
