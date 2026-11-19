"""
Explainability, Interpretability, and Credit Risk Decision Analysis
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import os
import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance


def compute_feature_importance(model, X_train, y_train, X_test, y_test, output_dir="outputs"):
    """
    Computes model feature importance using SHAP (TreeExplainer) or Permutation Importance.
    Returns:
        importance_df: DataFrame of ranked features and importance scores
        shap_values: Array of SHAP values if available, else None
        feature_names: Array of transformed feature names
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
        X_train_trans = X_train
        X_test_trans = X_test
        feature_names = X_train.columns.tolist()

    # Clean feature names for clean presentation
    clean_names = [
        str(f).replace("num__", "").replace("cat__", "").replace("_", " ").title()
        for f in feature_names
    ]

    shap_values = None
    mean_importance = None

    # Try SHAP
    try:
        import shap
        # Check if classifier is VotingClassifier
        if hasattr(classifier, "estimators_"):
            # TreeExplainer on XGBoost sub-estimator
            sub_est = classifier.estimators_[1] if len(classifier.estimators_) > 1 else classifier.estimators_[0]
            explainer = shap.TreeExplainer(sub_est)
            sv = explainer.shap_values(X_test_trans)
            if isinstance(sv, list):
                shap_values = sv[1]
            else:
                shap_values = sv
            mean_importance = np.mean(np.abs(shap_values), axis=0)
        elif hasattr(classifier, "feature_importances_"):
            explainer = shap.TreeExplainer(classifier)
            sv = explainer.shap_values(X_test_trans)
            if isinstance(sv, list):
                shap_values = sv[1]
            else:
                shap_values = sv
            mean_importance = np.mean(np.abs(shap_values), axis=0)
    except Exception as e:
        print(f"[Notice] SHAP TreeExplainer deferred: {e}. Utilizing Permutation Importance.")

    # Fallback / validation via Permutation Importance
    if mean_importance is None or np.all(mean_importance == 0):
        perm = permutation_importance(
            model, X_test, y_test, scoring="f1_macro", n_repeats=15, random_state=42
        )
        mean_importance = perm.importances_mean
        clean_names = [f.replace("_", " ").title() for f in X_test.columns.tolist()]

    importance_df = pd.DataFrame({
        "Feature": clean_names[:len(mean_importance)],
        "Importance": mean_importance
    }).sort_values(by="Importance", ascending=False).reset_index(drop=True)

    importance_df.to_csv(os.path.join(output_dir, "feature_importance.csv"), index=False)
    return importance_df, shap_values, clean_names, X_test_trans


def analyze_asymmetric_lending_costs(y_true, y_prob, cost_fp=5.0, cost_fn=1.0):
    """
    Computes total expected loss across classification thresholds [0.05, 0.95].
    In financial credit scoring (LO2):
    - False Positive (approving a bad loan that defaults) incurs heavy capital loss (e.g. 5x).
    - False Negative (rejecting a good borrower) incurs lost interest revenue (1x).
    Returns:
        thresholds: Array of evaluation thresholds
        total_costs: Computed portfolio losses
        optimal_threshold: Threshold minimizing monetary loss
    """
    thresholds = np.linspace(0.05, 0.95, 91)
    total_costs = []

    for t in thresholds:
        preds = (y_prob >= t).astype(int)
        fp = np.sum((preds == 1) & (y_true == 0))
        fn = np.sum((preds == 0) & (y_true == 1))
        cost = (cost_fp * fp) + (cost_fn * fn)
        total_costs.append(cost)

    total_costs = np.array(total_costs)
    opt_idx = np.argmin(total_costs)
    optimal_threshold = thresholds[opt_idx]

    return thresholds, total_costs, optimal_threshold
