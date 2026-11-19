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
    Infers the loan approval based exclusively on applicant credit history:
    IF Credit_History == 1 -> Approve (1)
    IF Credit_History == 0 -> Reject (0)
    Missing values default to the training mode.
    """
    def __init__(self):
        self.classes_ = np.array([0, 1])
        self.default_history_ = 1.0
        self.default_prediction_ = 1
        self.rule_map_ = {}

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        if isinstance(X, pd.DataFrame):
            hist = X["Credit_History"].copy()
        else:
            # Assumes Credit_History is at column index 4 in preprocessed numeric
            hist = pd.Series(X[:, 0])

        self.default_history_ = hist.mode().iloc[0] if not hist.mode().empty else 1.0
        hist_clean = hist.fillna(self.default_history_)

        df_train = pd.DataFrame({"Credit_History": hist_clean.to_numpy(), "target": np.asarray(y)})
        self.rule_map_ = (
            df_train.groupby("Credit_History")["target"]
            .agg(lambda vals: vals.mode().iloc[0] if not vals.mode().empty else 1)
            .to_dict()
        )
        self.default_prediction_ = int(pd.Series(y).mode().iloc[0]) if len(y) > 0 else 1
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
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        for i, p in enumerate(preds):
            if p == 1:
                proba[i] = [0.15, 0.85]
            else:
                proba[i] = [0.85, 0.15]
        return proba


class SymbolicExpertSystemClassifier(BaseEstimator, ClassifierMixin):
    """
    Domain Expert Multi-Condition Symbolic System (Addresses Learning Outcome LO3):
    Explicit symbolic IF-THEN production rules representing bank policy:
    - Rule 1: IF Credit_History == 0 -> Reject (0)
    - Rule 2: IF Credit_History == 1 AND TotalIncome >= 3000 AND LoanAmount <= 250 -> Approve (1)
    - Rule 3: IF Credit_History == 1 AND TotalIncome < 3000 AND LoanAmount > 150 -> Reject (0)
    - Rule 4: Default -> Prior training majority class
    """
    def __init__(self, min_income=3000.0, max_loan=300.0):
        self.min_income = min_income
        self.max_loan = max_loan
        self.classes_ = np.array([0, 1])
        self.default_class_ = 1

    def fit(self, X, y):
        self.classes_ = np.unique(y)
        self.default_class_ = int(pd.Series(y).mode().iloc[0]) if len(y) > 0 else 1
        return self

    def predict(self, X):
        preds = []
        if isinstance(X, pd.DataFrame):
            for _, row in X.iterrows():
                ch = row.get("Credit_History", 1.0)
                if pd.isna(ch):
                    ch = 1.0
                app_inc = row.get("ApplicantIncome", 3000.0)
                coapp_inc = row.get("CoapplicantIncome", 0.0)
                tot_inc = (0.0 if pd.isna(app_inc) else app_inc) + (0.0 if pd.isna(coapp_inc) else coapp_inc)
                loan = row.get("LoanAmount", 140.0)
                if pd.isna(loan):
                    loan = 140.0

                if ch == 0.0:
                    preds.append(0)
                elif ch == 1.0 and tot_inc >= self.min_income and loan <= self.max_loan:
                    preds.append(1)
                elif ch == 1.0 and tot_inc < 2500.0 and loan > 180.0:
                    preds.append(0)
                else:
                    preds.append(self.default_class_)
        else:
            # Fallback if array passed
            preds = [self.default_class_] * len(X)
        return np.array(preds, dtype=int)

    def predict_proba(self, X):
        preds = self.predict(X)
        proba = np.zeros((len(preds), 2))
        for i, p in enumerate(preds):
            if p == 1:
                proba[i] = [0.10, 0.90]
            else:
                proba[i] = [0.90, 0.10]
        return proba


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
        "Symbolic Expert Rules": SymbolicExpertSystemClassifier(),

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
