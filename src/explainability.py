"""
Explainability, Interpretability, and Leak-Free Credit Risk Decision Analysis
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import os
import re
import json
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def _format_feature_name(raw_name):
    s = str(raw_name).replace("num__", "").replace("cat__", "")
    # Insert space before camelCase capitals (e.g., LoanAmount -> Loan Amount)
    s = re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s)
    s = s.replace("_", " ").strip()
    return " ".join(w.capitalize() if not w.isupper() else w for w in s.split())


def _extract_positive_class_shap(sv):
    """Normalizes TreeExplainer output across SHAP versions to shape (n_samples, n_features)."""
    if isinstance(sv, list):
        return np.asarray(sv[1], dtype=float)
    arr = np.asarray(sv, dtype=float)
    if arr.ndim == 3:
        return arr[:, :, 1]
    return arr


def compute_feature_importance(model, X_train, y_train, X_test, y_test, output_dir="outputs"):
    """
    Computes BOTH:
      1. Exact Probability-Space Ensemble SHAP via weighted TreeExplainer linearity:
           phi_ens(x) = w_rf * phi_rf(x) + w_xgb * phi_xgb(x)
         where each sub-estimator is explained in predicted-probability space
         using the training split as the interventional background distribution.
      2. Model-Agnostic Permutation Importance on the full end-to-end ensemble
         pipeline (scoring='f1_macro', n_repeats=30) on raw input predictors.

    Returns:
        shap_df: DataFrame of transformed features ranked by Mean_Abs_SHAP (probability units) and SHAP_Share_Pct
        perm_df: DataFrame of raw features ranked by Perm_F1_Drop_Mean and Perm_F1_Drop_Std
        shap_values: (N_test, N_transformed_features) array of ensemble probability-space SHAP values
    """
    os.makedirs(output_dir, exist_ok=True)

    preprocessor = model.named_steps.get("preprocessor", None)
    classifier = model.named_steps.get("classifier", None)

    if preprocessor is not None:
        X_train_trans = preprocessor.transform(X_train)
        X_test_trans = preprocessor.transform(X_test)
        if hasattr(preprocessor, "get_feature_names_out"):
            feature_names = preprocessor.get_feature_names_out()
        else:
            feature_names = [f"feat_{i}" for i in range(X_train_trans.shape[1])]
    else:
        X_train_trans = np.asarray(X_train)
        X_test_trans = np.asarray(X_test)
        feature_names = X_train.columns.tolist()

    clean_names = [_format_feature_name(f) for f in feature_names]

    # 1. Exact Probability-Space SHAP across the full Soft-Voting Ensemble
    import shap
    shap_values = None
    if hasattr(classifier, "estimators_"):
        raw_weights = getattr(classifier, "weights", None)
        if raw_weights is None:
            weights = np.ones(len(classifier.estimators_), dtype=float)
        else:
            weights = np.asarray(raw_weights, dtype=float)
        weights = weights / np.sum(weights)

        combined_sv = np.zeros_like(X_test_trans, dtype=float)
        for est, w in zip(classifier.estimators_, weights):
            explainer = shap.TreeExplainer(
                est, data=X_train_trans, model_output="probability"
            )
            sv_est = _extract_positive_class_shap(
                explainer.shap_values(X_test_trans, check_additivity=False)
            )
            combined_sv += w * sv_est
        shap_values = combined_sv
    else:
        explainer = shap.TreeExplainer(
            classifier, data=X_train_trans, model_output="probability"
        )
        shap_values = _extract_positive_class_shap(
            explainer.shap_values(X_test_trans, check_additivity=False)
        )

    mean_abs_shap = np.mean(np.abs(shap_values), axis=0)
    total_shap = float(np.sum(mean_abs_shap)) if np.sum(mean_abs_shap) > 0 else 1.0
    shap_share_pct = 100.0 * mean_abs_shap / total_shap

    shap_df = pd.DataFrame({
        "Feature": clean_names[:len(mean_abs_shap)],
        "Mean_Abs_SHAP": mean_abs_shap,
        "SHAP_Share_Pct": shap_share_pct,
    }).sort_values(by="Mean_Abs_SHAP", ascending=False).reset_index(drop=True)
    shap_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)

    # 2. Full-Pipeline Model-Agnostic Permutation Importance (raw predictors)
    perm = permutation_importance(
        model, X_test, y_test, scoring="f1_macro", n_repeats=30, random_state=42
    )
    raw_clean_names = [_format_feature_name(f) for f in X_test.columns.tolist()]
    perm_df = pd.DataFrame({
        "Feature": raw_clean_names,
        "Perm_F1_Drop_Mean": perm.importances_mean,
        "Perm_F1_Drop_Std": perm.importances_std,
    }).sort_values(by="Perm_F1_Drop_Mean", ascending=False).reset_index(drop=True)
    perm_df.to_csv(os.path.join(output_dir, "permutation_importance.csv"), index=False)

    return shap_df, perm_df, shap_values


def _compute_cost_curve(y_true, y_prob, thresholds, cost_fp=5.0, cost_fn=1.0):
    """Computes raw portfolio penalty and normalized cost per 100 applications across thresholds."""
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    raw_costs = []
    fp_list = []
    fn_list = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        raw_costs.append((cost_fp * fp) + (cost_fn * fn))
        fp_list.append(fp)
        fn_list.append(fn)

    raw_costs = np.asarray(raw_costs, dtype=float)
    costs_per_100 = raw_costs * (100.0 / len(y_true))
    return raw_costs, costs_per_100, np.asarray(fp_list), np.asarray(fn_list)


def analyze_asymmetric_lending_costs_leak_free(
    y_train, oof_prob_train, y_test, prob_test, cost_fp=5.0, cost_fn=1.0, output_dir="outputs"
):
    """
    Leak-free asymmetric lending cost optimization (LO2):
      1. Sweeps decision thresholds tau in [0.05, 0.95] strictly on TRAINING
         out-of-fold (OOF) predicted probabilities to select tau*.
      2. Evaluates the frozen CV-selected tau* once on the independent held-out
         TEST set against the standard default cutoff tau = 0.50.
    """
    thresholds = np.linspace(0.05, 0.95, 91)

    oof_raw, oof_per_100, oof_fp, oof_fn = _compute_cost_curve(
        y_train, oof_prob_train, thresholds, cost_fp=cost_fp, cost_fn=cost_fn
    )
    opt_idx = int(np.argmin(oof_raw))
    opt_threshold = float(thresholds[opt_idx])
    default_idx = int(np.argmin(np.abs(thresholds - 0.50)))

    test_raw, test_per_100, test_fp, test_fn = _compute_cost_curve(
        y_test, prob_test, thresholds, cost_fp=cost_fp, cost_fn=cost_fn
    )

    summary = {
        "cost_fp": float(cost_fp),
        "cost_fn": float(cost_fn),
        "selected_threshold_oof": round(opt_threshold, 2),
        "oof_train": {
            "n_samples": int(len(y_train)),
            "default_tau_050_cost": float(oof_raw[default_idx]),
            "default_tau_050_cost_per_100": float(oof_per_100[default_idx]),
            "default_tau_050_fp": int(oof_fp[default_idx]),
            "default_tau_050_fn": int(oof_fn[default_idx]),
            "opt_tau_cost": float(oof_raw[opt_idx]),
            "opt_tau_cost_per_100": float(oof_per_100[opt_idx]),
            "opt_tau_fp": int(oof_fp[opt_idx]),
            "opt_tau_fn": int(oof_fn[opt_idx]),
            "cost_reduction_pct": float((1.0 - oof_raw[opt_idx] / oof_raw[default_idx]) * 100.0),
        },
        "heldout_test": {
            "n_samples": int(len(y_test)),
            "default_tau_050_cost": float(test_raw[default_idx]),
            "default_tau_050_cost_per_100": float(test_per_100[default_idx]),
            "default_tau_050_fp": int(test_fp[default_idx]),
            "default_tau_050_fn": int(test_fn[default_idx]),
            "opt_tau_cost": float(test_raw[opt_idx]),
            "opt_tau_cost_per_100": float(test_per_100[opt_idx]),
            "opt_tau_fp": int(test_fp[opt_idx]),
            "opt_tau_fn": int(test_fn[opt_idx]),
            "cost_reduction_pct": float((1.0 - test_raw[opt_idx] / test_raw[default_idx]) * 100.0),
        },
    }

    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
        with open(os.path.join(output_dir, "cost_threshold_analysis.json"), "w") as f:
            json.dump(summary, f, indent=4)

    return thresholds, oof_per_100, test_per_100, opt_threshold, summary
