"""
Master End-to-End Training, Cross-Validation, and Evaluation Pipeline
COMP1818 Artificial Intelligence Applications - University of Greenwich
Author: Kirtan Karkar
"""

import os
import shutil
import time
import pandas as pd
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
    analyze_asymmetric_lending_costs
)
from src.plotting import (
    plot_model_comparison,
    plot_feature_importance,
    plot_roc_pr_curves,
    plot_confusion_matrices,
    plot_cost_threshold_analysis,
    generate_architecture_diagram
)


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
    report_dir = os.path.join(base_dir, "report")

    os.makedirs(fig_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(report_dir, exist_ok=True)

    # 2. Data Loading & Preprocessing
    print("\n[Phase 1/5] Ingesting and Partitioning Dataset...")
    X_train, X_test, y_train, y_test, num_cols, cat_cols = get_data_splits(
        filepath=data_path, test_size=0.20, random_state=42
    )
    print(f"  • Total Applications: {len(X_train) + len(X_test)}")
    print(f"  • Training Records (80%): {len(X_train)} ({np.sum(y_train==1)} Approved, {np.sum(y_train==0)} Rejected)")
    print(f"  • Testing Records (20%): {len(X_test)} ({np.sum(y_test==1)} Approved, {np.sum(y_test==0)} Rejected)")
    print(f"  • Numerical Predictors: {num_cols}")
    print(f"  • Categorical Predictors: {cat_cols}")

    preprocessor = build_preprocessor(num_cols, cat_cols)

    # 3. Model Pipeline Construction
    print("\n[Phase 2/5] Building Baseline, Symbolic (LO3), and ML Model Pipelines...")
    models = build_model_pipelines(preprocessor, random_state=42)
    param_grids = get_hyperparameter_grids()
    print(f"  • Initialized {len(models)} model architectures across 4 paradigms:")
    print("    1. Trivial Statistical Baselines: Majority Class, Random Chance")
    print("    2. Symbolic Rule Baselines: Credit History 1-Rule, Expert Rules Engine (LO3)")
    print("    3. Supervised Machine Learning: Logistic Regression, Decision Tree, Random Forest, SVM, XGBoost")
    print("    4. Proposed Extension: Soft-Voting Ensemble (Random Forest + XGBoost)")

    # 4. Nested Cross-Validation Execution
    print("\n[Phase 3/5] Executing Nested Stratified 5-Fold Cross-Validation...")
    cv_summary, fold_scores, fitted_models = run_nested_cross_validation(
        models=models,
        param_grids=param_grids,
        X_train=X_train,
        y_train=y_train,
        outer_folds=5,
        inner_folds=3,
        random_state=42
    )

    # 5. Independent Test Set Evaluation
    print("\n[Phase 4/5] Evaluating Fitted Models on Independent Test Set ($N=123$)...")
    test_summary, preds_dict, probs_dict = evaluate_on_test_set(
        fitted_models=fitted_models,
        X_test=X_test,
        y_test=y_test
    )

    # Save and merge tabular results
    results_df = merge_and_save_results(
        cv_df=cv_summary,
        test_df=test_summary,
        fold_df=fold_scores,
        output_dir=out_dir
    )

    # 6. Explainability, Interpretability & Asymmetric Lending Costs
    print("\n[Phase 5/5] Computing Interpretability & Risk Cost Optimisation...")
    ensemble_model = fitted_models["RF + XGBoost Soft Voting"]
    importance_df, shap_vals, clean_feat_names, X_test_trans = compute_feature_importance(
        model=ensemble_model,
        X_train=X_train,
        y_train=y_train,
        X_test=X_test,
        y_test=y_test,
        output_dir=out_dir
    )

    ens_probs = probs_dict["RF + XGBoost Soft Voting"]
    thresholds, costs, opt_threshold = analyze_asymmetric_lending_costs(
        y_true=y_test.to_numpy(),
        y_prob=ens_probs,
        cost_fp=5.0,  # Bad loan defaulted
        cost_fn=1.0   # Creditworthy borrower rejected
    )
    print(f"  • Asymmetric Lending Risk Analysis:")
    print(f"    - Default Cutoff (τ = 0.50) Loss: {costs[np.argmin(np.abs(thresholds - 0.50))]:.1f}")
    print(f"    - Optimal Decision Cutoff (τ* = {opt_threshold:.2f}) Loss: {np.min(costs):.1f}")
    print(f"    - Capital Loss Reduction: {(1.0 - np.min(costs)/costs[np.argmin(np.abs(thresholds - 0.50))])*100:.1f}%")

    # 7. Generating Publication-Quality Figures
    print("\nGenerating Figures...")
    p1 = plot_model_comparison(results_df, output_dir=fig_dir)
    p2 = plot_feature_importance(importance_df, top_n=10, output_dir=fig_dir)
    p3 = plot_roc_pr_curves(probs_dict, y_test, output_dir=fig_dir)
    p4 = plot_confusion_matrices(preds_dict, y_test, output_dir=fig_dir)
    p5 = plot_cost_threshold_analysis(thresholds, costs, opt_threshold, output_dir=fig_dir)
    p6 = generate_architecture_diagram(output_dir=fig_dir)

    # Copy all figures into the report directory for Overleaf compilation
    for fig_file in [p1, p2, p3, p4, p5, p6]:
        shutil.copy(fig_file, report_dir)

    elapsed = time.time() - start_time
    print("\n" + "=" * 80)
    print(f" EXECUTION COMPLETE (Runtime: {elapsed:.2f} seconds)")
    print("=" * 80)
    print("\nQUANTITATIVE BENCHMARK SUMMARY:")
    display_cols = ["Model", "Test Accuracy", "Test Macro F1", "Test ROC AUC", "CV Macro F1 Mean", "CV Macro F1 Std"]
    print(results_df[display_cols].to_string(index=False))

    print("\nArtifacts and deliverables generated:")
    print(f"  • Tabular Results: {os.path.join(out_dir, 'evaluation_results.csv')}")
    print(f"  • LaTeX Table Code: {os.path.join(out_dir, 'latex_results_table.tex')}")
    print(f"  • Publication Figures: {fig_dir}/")
    print(f"  • Overleaf Report Assets: {report_dir}/")


if __name__ == "__main__":
    main()
