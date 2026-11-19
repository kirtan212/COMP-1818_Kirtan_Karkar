"""
Evaluation and Nested Cross-Validation Engine
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score
)


def run_nested_cross_validation(
    models, param_grids, X_train, y_train, outer_folds=5, inner_folds=3, random_state=42
):
    """
    Executes nested stratified cross-validation (5x3) to estimate generalisation
    without data leakage, and records out-of-fold (OOF) scores on X_train for
    leak-free downstream decision threshold selection.
    Outer CV: Estimates test error variance across unseen splits.
    Inner CV: Tunes hyperparameters using StratifiedKFold on Macro F1.
    """
    outer_cv = StratifiedKFold(n_splits=outer_folds, shuffle=True, random_state=random_state)
    inner_cv = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=random_state)

    results_summary = []
    fold_scores_list = []
    fitted_models = {}
    oof_probs_dict = {}

    for name, pipeline in models.items():
        print(f"--> Evaluating: {name}")
        outer_acc_scores = []
        outer_f1_scores = []
        outer_roc_scores = []
        oof_scores = np.full(len(X_train), np.nan, dtype=float)

        has_params = name in param_grids and len(param_grids[name]) > 0

        for fold_idx, (train_idx, val_idx) in enumerate(outer_cv.split(X_train, y_train)):
            X_tr, y_tr = X_train.iloc[train_idx], y_train.iloc[train_idx]
            X_vl, y_vl = X_train.iloc[val_idx], y_train.iloc[val_idx]

            if has_params:
                grid = GridSearchCV(
                    estimator=clone(pipeline),
                    param_grid=param_grids[name],
                    scoring="f1_macro",
                    cv=inner_cv,
                    n_jobs=-1,
                    refit=True
                )
                grid.fit(X_tr, y_tr)
                estimator = grid.best_estimator_
            else:
                estimator = clone(pipeline)
                estimator.fit(X_tr, y_tr)

            preds = estimator.predict(X_vl)
            acc = accuracy_score(y_vl, preds)
            f1_mac = f1_score(y_vl, preds, average="macro", zero_division=0)

            roc = np.nan
            if hasattr(estimator, "predict_proba"):
                try:
                    prob = estimator.predict_proba(X_vl)[:, 1]
                    oof_scores[val_idx] = prob
                    roc = roc_auc_score(y_vl, prob)
                except Exception:
                    pass
            elif hasattr(estimator, "decision_function"):
                try:
                    scores = estimator.decision_function(X_vl)
                    oof_scores[val_idx] = scores
                    roc = roc_auc_score(y_vl, scores)
                except Exception:
                    pass

            outer_acc_scores.append(acc)
            outer_f1_scores.append(f1_mac)
            if not np.isnan(roc):
                outer_roc_scores.append(roc)

            fold_scores_list.append({
                "Model": name,
                "Outer_Fold": fold_idx + 1,
                "Val_Accuracy": acc,
                "Val_Macro_F1": f1_mac,
                "Val_ROC_AUC": roc
            })

        if not np.all(np.isnan(oof_scores)):
            oof_probs_dict[name] = oof_scores

        # Fit final model on the entire training set
        if has_params:
            final_search = GridSearchCV(
                estimator=clone(pipeline),
                param_grid=param_grids[name],
                scoring="f1_macro",
                cv=inner_cv,
                n_jobs=-1,
                refit=True
            )
            final_search.fit(X_train, y_train)
            best_model = final_search.best_estimator_
            best_params = final_search.best_params_
        else:
            best_model = clone(pipeline)
            best_model.fit(X_train, y_train)
            best_params = {}

        fitted_models[name] = best_model

        results_summary.append({
            "Model": name,
            "CV Accuracy Mean": float(np.mean(outer_acc_scores)),
            "CV Accuracy Std": float(np.std(outer_acc_scores, ddof=1) if len(outer_acc_scores) > 1 else 0.0),
            "CV Macro F1 Mean": float(np.mean(outer_f1_scores)),
            "CV Macro F1 Std": float(np.std(outer_f1_scores, ddof=1) if len(outer_f1_scores) > 1 else 0.0),
            "CV ROC AUC Mean": float(np.mean(outer_roc_scores)) if len(outer_roc_scores) > 0 else np.nan,
            "Best Params": best_params
        })

    cv_df = pd.DataFrame(results_summary)
    fold_df = pd.DataFrame(fold_scores_list)
    return cv_df, fold_df, fitted_models, oof_probs_dict


def evaluate_on_test_set(fitted_models, X_test, y_test):
    """
    Evaluates final fitted models on independent held-out test split (20%, N=123).
    Computes comprehensive binary and macro-averaged metrics.
    """
    test_metrics = []
    predictions_dict = {}
    probabilities_dict = {}

    for name, model in fitted_models.items():
        preds = model.predict(X_test)
        predictions_dict[name] = preds

        acc = accuracy_score(y_test, preds)
        prec_mac = precision_score(y_test, preds, average="macro", zero_division=0)
        rec_mac = recall_score(y_test, preds, average="macro", zero_division=0)
        f1_mac = f1_score(y_test, preds, average="macro", zero_division=0)

        rec_reject = recall_score(y_test, preds, pos_label=0, zero_division=0)
        rec_approve = recall_score(y_test, preds, pos_label=1, zero_division=0)
        prec_reject = precision_score(y_test, preds, pos_label=0, zero_division=0)
        prec_approve = precision_score(y_test, preds, pos_label=1, zero_division=0)

        roc = np.nan
        pr_auc = np.nan

        if hasattr(model, "predict_proba"):
            try:
                prob = model.predict_proba(X_test)[:, 1]
                roc = roc_auc_score(y_test, prob)
                pr_auc = average_precision_score(y_test, prob)
                probabilities_dict[name] = prob
            except Exception:
                pass
        elif hasattr(model, "decision_function"):
            try:
                scores = model.decision_function(X_test)
                roc = roc_auc_score(y_test, scores)
                pr_auc = average_precision_score(y_test, scores)
                probabilities_dict[name] = scores
            except Exception:
                pass

        test_metrics.append({
            "Model": name,
            "Test Accuracy": float(acc),
            "Test Macro Precision": float(prec_mac),
            "Test Macro Recall": float(rec_mac),
            "Test Macro F1": float(f1_mac),
            "Test ROC AUC": float(roc) if not np.isnan(roc) else np.nan,
            "Test PR AUC": float(pr_auc) if not np.isnan(pr_auc) else np.nan,
            "Recall Reject (0)": float(rec_reject),
            "Recall Approve (1)": float(rec_approve),
            "Precision Reject (0)": float(prec_reject),
            "Precision Approve (1)": float(prec_approve)
        })

    test_df = pd.DataFrame(test_metrics)
    return test_df, predictions_dict, probabilities_dict


def merge_and_save_results(cv_df, test_df, fold_df, output_dir="outputs"):
    """
    Merges CV and Test evaluations, writes CSV, JSON, and LaTeX tables.
    """
    os.makedirs(output_dir, exist_ok=True)

    combined_df = pd.merge(test_df, cv_df, on="Model")
    combined_df.to_csv(os.path.join(output_dir, "evaluation_results.csv"), index=False)
    fold_df.to_csv(os.path.join(output_dir, "cv_fold_scores.csv"), index=False)

    with open(os.path.join(output_dir, "evaluation_results.json"), "w") as f:
        json.dump(combined_df.to_dict(orient="records"), f, indent=4)

    latex_table = generate_latex_table(combined_df)
    with open(os.path.join(output_dir, "latex_results_table.tex"), "w") as f:
        f.write(latex_table)

    return combined_df


def generate_latex_table(results_df):
    """
    Generates a clean, publication-ready LaTeX table using booktabs.
    """
    lines = [
        r"\begin{table*}[t]",
        r"\centering",
        r"\caption{Performance comparison of trivial baselines, symbolic rule models, and machine-learning classifiers on the held-out test set ($N=123$) and nested stratified $5\times 3$ cross-validation ($N=491$). Values in parentheses indicate sample standard deviation across the five outer CV folds.}",
        r"\label{tab:results}",
        r"\small",
        r"\begin{tabular}{lccccccc}",
        r"\toprule",
        r"\textbf{Model Architecture} & \textbf{Test Acc.} & \textbf{Test Macro F1} & \textbf{Test ROC-AUC} & \textbf{Reject Recall} & \textbf{Approve Recall} & \textbf{CV Macro F1 ($\pm \sigma$)} & \textbf{CV Acc. ($\pm \sigma$)} \\",
        r"\midrule",
        r"\multicolumn{8}{l}{\textit{Trivial \& Symbolic Baselines}} \\"
    ]

    for _, row in results_df.iterrows():
        m_name = row["Model"]
        if m_name == "Logistic Regression":
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{Supervised Machine Learning Classifiers}} \\")

        test_acc = f"{row['Test Accuracy']:.3f}"
        test_f1 = f"{row['Test Macro F1']:.3f}"
        test_roc = f"{row['Test ROC AUC']:.3f}" if not np.isnan(row['Test ROC AUC']) else "--"
        rec_rej = f"{row['Recall Reject (0)']:.3f}"
        rec_app = f"{row['Recall Approve (1)']:.3f}"
        cv_f1 = f"{row['CV Macro F1 Mean']:.3f} ($\\pm${row['CV Macro F1 Std']:.3f})"
        cv_acc = f"{row['CV Accuracy Mean']:.3f} ($\\pm${row['CV Accuracy Std']:.3f})"

        if "Soft Voting" in m_name:
            lines.append(r"\midrule")
            lines.append(r"\multicolumn{8}{l}{\textit{Proposed Extension: Soft-Voting Ensemble (Gawande et al. 2026 adaptation)}} \\")
            line = f"\\textbf{{{m_name}}} & {test_acc} & {test_f1} & {test_roc} & {rec_rej} & {rec_app} & {cv_f1} & {cv_acc} \\\\"
        elif m_name == "Symbolic Expert Rules":
            line = f"\\textbf{{{m_name}}} & \\textbf{{{test_acc}}} & \\textbf{{{test_f1}}} & {test_roc} & \\textbf{{{rec_rej}}} & {rec_app} & \\textbf{{{cv_f1}}} & \\textbf{{{cv_acc}}} \\\\"
        elif m_name == "Logistic Regression":
            line = f"{m_name} & {test_acc} & {test_f1} & \\textbf{{{test_roc}}} & {rec_rej} & {rec_app} & {cv_f1} & {cv_acc} \\\\"
        else:
            line = f"{m_name} & {test_acc} & {test_f1} & {test_roc} & {rec_rej} & {rec_app} & {cv_f1} & {cv_acc} \\\\"

        lines.append(line)

    lines.extend([
        r"\bottomrule",
        r"\end{tabular}",
        r"\end{table*}"
    ])

    return "\n".join(lines)
