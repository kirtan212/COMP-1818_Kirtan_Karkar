"""
Model Definitions and Architectures for Loan Approval Prediction
COMP1818 Artificial Intelligence Applications - University of Greenwich
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import Pipeline
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier


# ==============================================================================
# 1. TRIVIAL & SYMBOLIC BASELINES
# ==============================================================================

class CreditHistoryRuleClassifier(BaseEstimator, ClassifierMixin):
    """
    Symbolic 1-Rule Classifier:
    Infers loan approval exclusively from applicant credit history:
      - Rule 1: IF Credit_History == 0 -> Reject (0)
      - Rule 2: IF Credit_History == 1 -> Approve (1)
    Missing Credit_History values are imputed with the training-fold mode.
    Class probabilities reflect empirical training-fold approval frequencies.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y_arr)
        if isinstance(X, pd.DataFrame):
            hist = X["Credit_History"].copy()
        else:
            hist = pd.Series(X[:, 0])

        self.default_history_ = float(hist.mode().iloc[0]) if not hist.mode().empty else 1.0
        hist_clean = hist.fillna(self.default_history_)

        df_train = pd.DataFrame({"Credit_History": hist_clean.to_numpy(), "target": y_arr})
        self.rule_map_ = (
            df_train.groupby("Credit_History")["target"]
            .agg(lambda vals: int(vals.mode().iloc[0]) if not vals.mode().empty else 1)
            .to_dict()
        )
        self.rule_prob_map_ = (
            df_train.groupby("Credit_History")["target"].mean().to_dict()
        )
        self.default_prediction_ = int(pd.Series(y_arr).mode().iloc[0]) if len(y_arr) > 0 else 1
        self.default_prob_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5
        return self

    def predict(self, X):
        if isinstance(X, pd.DataFrame):
            hist = X["Credit_History"].copy()
        else:
            hist = pd.Series(X[:, 0])

        hist_clean = hist.fillna(self.default_history_)
        preds = hist_clean.map(self.rule_map_).fillna(self.default_prediction_)
        return preds.to_numpy().astype(int)

    def predict_proba(self, X):
        if isinstance(X, pd.DataFrame):
            hist = X["Credit_History"].copy()
        else:
            hist = pd.Series(X[:, 0])

        hist_clean = hist.fillna(self.default_history_)
        p1 = hist_clean.map(self.rule_prob_map_).fillna(self.default_prob_).to_numpy(dtype=float)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])


class SymbolicExpertSystemClassifier(BaseEstimator, ClassifierMixin):
    """
    Domain Expert Multi-Condition Symbolic System (Addresses Learning Outcome LO3):
    Explicit propositional IF-THEN underwriting rules combining credit history
    with a standard Loan-to-Annual-Income (LTI) affordability ceiling:
      Let LTI = (1000 * LoanAmount) / (12 * (ApplicantIncome + CoapplicantIncome))
      - Rule 1 (Credit History Gate):  IF Credit_History == 0 -> Reject (0)
      - Rule 2 (Affordability Cap):    IF Credit_History == 1 AND LTI > max_lti -> Reject (0)
      - Rule 3 (Eligible Approval):    IF Credit_History == 1 AND LTI <= max_lti -> Approve (1)
    Missing Credit_History and LoanAmount values are imputed using training-fold
    mode and median learned strictly during fit().
    """
    def __init__(self, max_lti=3.0):
        self.max_lti = max_lti
        self.classes_ = np.array([0, 1])

    def _evaluate_rules(self, X):
        """Returns integer rule IDs (1, 2, or 3) fired for each row."""
        if not isinstance(X, pd.DataFrame):
            return np.full(len(X), 3, dtype=int)

        ch = X["Credit_History"].fillna(self.default_history_).to_numpy(dtype=float)
        app_inc = X["ApplicantIncome"].fillna(0.0).to_numpy(dtype=float)
        coapp_inc = X["CoapplicantIncome"].fillna(0.0).to_numpy(dtype=float)
        tot_inc = np.maximum(app_inc + coapp_inc, 1.0)
        loan = X["LoanAmount"].fillna(self.median_loan_).to_numpy(dtype=float)

        # LoanAmount is recorded in thousands; incomes are monthly
        lti = (loan * 1000.0) / (12.0 * tot_inc)

        rule_ids = np.full(len(X), 3, dtype=int)
        rule_ids[(ch == 1.0) & (lti > self.max_lti)] = 2
        rule_ids[ch == 0.0] = 1
        return rule_ids

    def fit(self, X, y):
        y_arr = np.asarray(y, dtype=int)
        self.classes_ = np.unique(y_arr)
        self.default_class_ = int(pd.Series(y_arr).mode().iloc[0]) if len(y_arr) > 0 else 1
        self.default_prob_ = float(np.mean(y_arr)) if len(y_arr) > 0 else 0.5

        if isinstance(X, pd.DataFrame):
            self.default_history_ = (
                float(X["Credit_History"].mode().iloc[0])
                if not X["Credit_History"].mode().empty else 1.0
            )
            self.median_loan_ = (
                float(X["LoanAmount"].median())
                if not X["LoanAmount"].dropna().empty else 128.0
            )
        else:
            self.default_history_ = 1.0
            self.median_loan_ = 128.0

        rule_ids = self._evaluate_rules(X)
        self.rule_prob_map_ = {}
        for rid in (1, 2, 3):
            mask = rule_ids == rid
            if np.any(mask):
                self.rule_prob_map_[rid] = float(np.mean(y_arr[mask]))
            else:
                self.rule_prob_map_[rid] = 0.0 if rid in (1, 2) else 1.0
        return self

    def count_rule_firings(self, X):
        """Returns a dictionary with counts of how many rows trigger each rule."""
        rule_ids = self._evaluate_rules(X)
        return {
            "Rule 1 (CH == 0 -> Reject)": int(np.sum(rule_ids == 1)),
            f"Rule 2 (CH == 1 & LTI > {self.max_lti:.1f} -> Reject)": int(np.sum(rule_ids == 2)),
            f"Rule 3 (CH == 1 & LTI <= {self.max_lti:.1f} -> Approve)": int(np.sum(rule_ids == 3)),
        }

    def predict(self, X):
        rule_ids = self._evaluate_rules(X)
        return np.where(rule_ids == 3, 1, 0).astype(int)

    def predict_proba(self, X):
        rule_ids = self._evaluate_rules(X)
        p1 = np.array([self.rule_prob_map_.get(int(r), self.default_prob_) for r in rule_ids], dtype=float)
        p1 = np.clip(p1, 1e-6, 1.0 - 1e-6)
        return np.column_stack([1.0 - p1, p1])


# ==============================================================================
# 2. MODEL FACTORY AND PIPELINES
# ==============================================================================

def build_model_pipelines(preprocessor, random_state=42):
    """
    Constructs all baseline and candidate ML model pipelines.
    Returns:
        dict of {model_name: pipeline}
    """
    pipelines = {
        # Trivial Baselines
        "Majority Baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="most_frequent"))
        ]),
        "Random Baseline": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DummyClassifier(strategy="stratified", random_state=random_state))
        ]),

        # Symbolic Rule Baselines (LO3)
        "Credit History One-Rule": CreditHistoryRuleClassifier(),
        "Symbolic Expert Rules": SymbolicExpertSystemClassifier(max_lti=3.0),

        # Machine Learning Classifiers
        "Logistic Regression": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=3000, random_state=random_state))
        ]),
        "Decision Tree": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", DecisionTreeClassifier(random_state=random_state))
        ]),
        "Random Forest": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", RandomForestClassifier(random_state=random_state, n_jobs=-1))
        ]),
        "Support Vector Machine": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", SVC(kernel="rbf", random_state=random_state))
        ]),
        "XGBoost": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", XGBClassifier(
                eval_metric="logloss",
                random_state=random_state,
                n_jobs=-1,
                tree_method="hist"
            ))
        ]),

        # Proposed Extension: Soft-Voting Ensemble (Gawande et al. 2026 adaptation)
        "RF + XGBoost Soft Voting": Pipeline([
            ("preprocessor", preprocessor),
            ("classifier", VotingClassifier(
                estimators=[
                    ("rf", RandomForestClassifier(n_estimators=150, max_depth=6, random_state=random_state, n_jobs=-1)),
                    ("xgb", XGBClassifier(n_estimators=100, max_depth=3, learning_rate=0.08, eval_metric="logloss", random_state=random_state, n_jobs=-1))
                ],
                voting="soft",
                weights=[1, 1]
            ))
        ])
    }
    return pipelines


def get_hyperparameter_grids():
    """
    Returns search grids for hyperparameter optimization during nested CV.
    Targeted, sensible parameter spaces avoiding combinatorial explosion.
    """
    grids = {
        "Logistic Regression": {
            "classifier__C": [0.05, 0.2, 1.0, 5.0],
            "classifier__class_weight": [None, "balanced"],
            "classifier__solver": ["liblinear", "lbfgs"]
        },
        "Decision Tree": {
            "classifier__max_depth": [2, 3, 5, 8],
            "classifier__min_samples_split": [2, 5, 10],
            "classifier__min_samples_leaf": [1, 3, 5],
            "classifier__class_weight": [None, "balanced"]
        },
        "Random Forest": {
            "classifier__n_estimators": [100, 180],
            "classifier__max_depth": [3, 5, 8, None],
            "classifier__min_samples_leaf": [1, 3],
            "classifier__class_weight": [None, "balanced"]
        },
        "Support Vector Machine": {
            "classifier__C": [0.1, 1.0, 5.0],
            "classifier__gamma": ["scale", "auto"],
            "classifier__class_weight": [None, "balanced"]
        },
        "XGBoost": {
            "classifier__n_estimators": [60, 120],
            "classifier__max_depth": [2, 3, 5],
            "classifier__learning_rate": [0.03, 0.08, 0.15],
            "classifier__subsample": [0.8, 1.0]
        },
        "RF + XGBoost Soft Voting": {
            "classifier__weights": [[1, 1], [1, 2], [2, 1]]
        }
    }
    return grids
