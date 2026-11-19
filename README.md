# COMP1818 Artificial Intelligence Applications - Coursework Project
## Explainable Ensemble Learning and Symbolic Baseline Benchmarking for Retail Loan Approval Prediction

**Institution:** School of Computing and Mathematical Sciences, University of Greenwich, London, UK  
**Module:** COMP1818 Artificial Intelligence Applications  
**Module Leader:** Hooman Oroojeni  
**Student Author:** Kirtan Karkar  
**Anchor Paper:** Gawande et al. (2026) *An Explainable Ensemble Learning Framework for Transparent and Accurate Loan Approval Prediction*, Discover Artificial Intelligence, 6(1), 1070. DOI: `10.1007/s44163-026-01569-6`  
**Dataset:** Kaggle Loan Prediction Problem Dataset ($N=614$, 11 predictors)

---

### Executive Overview & Alignment with Learning Outcomes

1. **LO1 (Emergence of AI in Credit Systems):** Investigates the transition from traditional credit scorecards to modern non-linear ML ensembles, contrasting our $N=614$ benchmark against Gawande et al. (2026) ($N=13{,}508$, 95.16% accuracy) to demonstrate how sample size governs model complexity gains.
2. **LO2 (Real-World System Design, Leak-Free Data Use & Asymmetric Lending Costs):** Implements leak-free `ColumnTransformer` pipelines, nested stratified $5 \times 3$ cross-validation, saved trained models (`models/`), and leak-free out-of-fold (OOF) asymmetric loss threshold tuning ($C_{\text{FP}} = 5 \cdot C_{\text{FN}}$). Selecting $\tau^* = 0.84$ strictly on training OOF probabilities yields a verified **5.7%** out-of-sample risk-cost reduction on the untouched test set ($88.0 \to 83.0$, reducing false approvals from 17 to 4).
3. **LO3 (Symbolic AI vs. Statistical Machine Learning):** Benchmarks trivial baselines (Majority Class, Stratified Random) and explicit propositional symbolic rules (Credit History 1-Rule and a 3-clause Symbolic Expert Rules Engine adding a $3.0\times$ Loan-to-Annual-Income affordability cap) against five tuned ML classifiers and an RF + XGBoost Soft-Voting Ensemble, explained via both exact probability-space ensemble SHAP and 30-repeat full-pipeline permutation importance.

---

### Project Directory Structure

```text
comp1818_loan_approval_ai/
├── data/
│   └── train.csv                               # Canonical Kaggle Loan Prediction Dataset (614 records)
├── figures/
│   ├── figure01.png                            # System methodological architecture diagram
│   ├── model_comparison.png                    # Nested 5x3 CV Macro F1 vs Test Macro F1 bar chart
│   ├── confusion_matrices.png                  # 2x2 grid of normalized test confusion matrices
│   ├── roc_pr_curves.png                       # Side-by-side ROC and Precision-Recall curves
│   ├── feature_importance.png                  # Ensemble probability-space SHAP & permutation importance
│   └── calibration_cost_tradeoff.png           # Leak-free OOF & test asymmetric cost curves (LO2)
├── models/
│   ├── *.joblib                                # Saved trained pipelines for all 10 benchmarked models
│   └── model_manifest.json                     # Manifest of saved models, best hyperparameters, and OOF tau*
├── notebooks/
│   └── comp1818_loan_approval_experiment.ipynb # Step-by-step interactive Jupyter notebook
├── outputs/
│   ├── evaluation_results.csv                  # Quantitative benchmark metrics (CV + held-out test)
│   ├── evaluation_results.json                 # Full raw JSON metrics
│   ├── cv_fold_scores.csv                      # Per-fold outer CV metrics across all 5 outer folds
│   ├── feature_importance.csv                  # Exact probability-space ensemble SHAP attributions
│   ├── permutation_importance.csv              # Full-pipeline permutation importance (30 repeats)
│   ├── cost_threshold_analysis.json            # Leak-free OOF selection and held-out test cost audit
│   └── latex_results_table.tex                 # Formatted booktabs LaTeX table code
├── report/
│   ├── COMPXXXX.cls                            # University of Greenwich official LaTeX class
│   ├── COMPXXXX.bst                            # University of Greenwich bibliography style
│   ├── references_AI.bib                       # Verified BibTeX citations (all DOIs Crossref-checked)
│   ├── karkar_kirtan_AI_Intro.tex              # 4-page academic report + AI declaration & references
│   └── (4 report figures copied here)          # Self-contained for Overleaf pdfLaTeX compilation
├── src/
│   ├── __init__.py                             # Package initialization
│   ├── data_loader.py                          # Data ingestion, target mapping, leak-free pipelines
│   ├── models.py                               # Trivial baselines, symbolic rule engines, and ML models
│   ├── evaluate.py                             # Nested 5x3 CV engine, OOF capture, test evaluator
│   ├── explainability.py                       # Ensemble TreeSHAP, permutation importance, OOF cost tuning
│   └── plotting.py                             # Publication-grade 300-DPI figures
├── requirements.txt                            # Python dependencies
├── train_and_evaluate.py                       # Master end-to-end execution script
└── README.md                                   # Project documentation and reproduction guide
```

---

### Quantitative Benchmark Results

Evaluated across $N_{\text{train}}=491$ (Nested Stratified $5 \times 3$ Cross-Validation) and $N_{\text{test}}=123$ (Independent Held-Out Test Set):

| Model Architecture | Test Accuracy | Test Macro F1 | Test ROC-AUC | Reject Recall (0) | Approve Recall (1) | Nested CV Macro F1 | Nested CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.691 | 0.409 | 0.500 | 0.000 | 1.000 | 0.407 ($\pm$0.002) | 0.686 ($\pm$0.004) |
| **Random Baseline (Stratified)** | 0.553 | 0.480 | 0.480 | 0.289 | 0.671 | 0.505 ($\pm$0.067) | 0.576 ($\pm$0.059) |
| **Credit History One-Rule** | 0.854 | 0.802 | 0.770 | 0.553 | 0.988 | 0.707 ($\pm$0.089) | 0.798 ($\pm$0.051) |
| **Symbolic Expert Rules (LO3)** | **0.886** | **0.858** | 0.843 | **0.711** | 0.965 | **0.728 ($\pm$0.103)** | **0.804 ($\pm$0.061)** |
| **Logistic Regression** | 0.862 | 0.815 | **0.849** | 0.579 | 0.988 | 0.711 ($\pm$0.089) | 0.800 ($\pm$0.051) |
| **Decision Tree (CART)** | 0.854 | 0.802 | 0.757 | 0.553 | 0.988 | 0.693 ($\pm$0.081) | 0.786 ($\pm$0.043) |
| **Random Forest** | 0.837 | 0.784 | 0.795 | 0.553 | 0.965 | 0.668 ($\pm$0.085) | 0.747 ($\pm$0.074) |
| **Support Vector Machine (SVM)** | 0.846 | 0.788 | 0.826 | 0.526 | 0.988 | 0.681 ($\pm$0.080) | 0.764 ($\pm$0.055) |
| **XGBoost Classifier** | 0.854 | 0.802 | 0.795 | 0.553 | 0.988 | 0.697 ($\pm$0.100) | 0.788 ($\pm$0.058) |
| **RF + XGBoost Soft Voting** | 0.837 | 0.784 | 0.780 | 0.553 | 0.965 | 0.692 ($\pm$0.083) | 0.784 ($\pm$0.049) |

---

### Step-by-Step Reproduction Instructions

#### 1. Install Dependencies
From the repository root, install the required Python packages (recommended inside a virtual environment):
```bash
pip install -r requirements.txt
```

#### 2. Run the End-to-End Benchmark Pipeline
Replicates all nested $5 \times 3$ CV runs, evaluates the held-out test set, computes exact probability-space ensemble SHAP and 30-repeat permutation importance, performs leak-free OOF asymmetric cost tuning, saves all 10 trained models to `models/`, and regenerates every table and figure:
```bash
python3 train_and_evaluate.py
```
*Expected runtime: ~25–35 seconds.*

#### 3. Interactive Jupyter Notebook
```bash
jupyter notebook notebooks/comp1818_loan_approval_experiment.ipynb
```

#### 4. Compiling the Report on Overleaf
1. Zip the contents of `report/` locally (do not commit the zip archive to Git) and upload it to [Overleaf](https://www.overleaf.com).
2. Set the main document to `karkar_kirtan_AI_Intro.tex` and compile with **pdfLaTeX**.
3. Append the official Moodle **Declaration of AI Use** page to the exported PDF prior to final submission.
