# COMP1818 Artificial Intelligence Applications - Coursework Project
## Explainable Ensemble Learning and Symbolic Baseline Benchmarking for Retail Loan Approval Prediction

**Institution:** School of Computing and Mathematical Sciences, University of Greenwich, London, UK  
**Module:** COMP1818 Artificial Intelligence Applications  
**Module Leader:** Hooman Oroojeni  
**Student Author:** Kirtan Karkar  
**Anchor Paper:** Gawande et al. (2026) *An Explainable Ensemble Learning Framework for Transparent and Accurate Loan Approval Prediction*, Discover Artificial Intelligence, Springer.  
**Dataset:** Kaggle Loan Prediction Problem Dataset ($N=614$)

---

### Executive Overview & Alignment with Learning Outcomes

1. **LO1 (Emergence of AI in Credit Systems):** Investigates the transition from traditional linear credit scorecards to modern non-linear machine learning ensembles, identifying the boundaries of statistical learning on modest-sized financial datasets ($N=614$).
2. **LO2 (Real-World System Design, Use of Data & Asymmetric Lending Costs):** Evaluates end-to-end data pipelines with leak-free preprocessing (`ColumnTransformer`), nested stratified cross-validation ($5 \times 3$), and formulates an asymmetric loss function ($C_{\text{FP}} = 5 \cdot C_{\text{FN}}$). Optimizing the classification threshold from $\tau=0.50$ to $\tau^*=0.82$ reduces institutional default risk by **15.9%**.
3. **LO3 (Symbolic AI vs. Statistical Machine Learning):** Critically benchmarks simple propositional rule heuristics (Credit History 1-Rule and a multi-condition Symbolic Expert Rule Engine) against statistical machine learning (Logistic Regression, Decision Trees, Random Forests, SVM, XGBoost, and an Explainable Soft-Voting Ensemble). Demonstrates that while symbolic systems provide deterministic compliance with the UK/EU GDPR Article 22 and the US Fair Credit Reporting Act (FCRA), statistical ensembles capture continuous multi-attribute trade-offs.

---

### Project Directory Structure

```text
comp1818_loan_approval_ai/
├── data/
│   └── train.csv                            # Official Kaggle Loan Prediction Dataset (614 records)
├── figures/
│   ├── figure01.png                         # System methodological architecture diagram
│   ├── model_comparison.png                 # Nested CV Macro F1 vs Test Macro F1 bar chart
│   ├── confusion_matrices.png               # 2x2 grid of normalized test confusion matrices
│   ├── roc_pr_curves.png                    # Side-by-side ROC and Precision-Recall curves
│   ├── feature_importance.png               # Global SHAP & permutation feature attribution
│   └── calibration_cost_tradeoff.png        # Asymmetric risk cost curve & optimal threshold (LO2)
├── notebooks/
│   └── comp1818_loan_approval_experiment.ipynb # Step-by-step interactive Jupyter notebook
├── outputs/
│   ├── evaluation_results.csv               # Comprehensive quantitative benchmark metrics
│   ├── evaluation_results.json              # Full raw JSON metrics
│   ├── cv_fold_scores.csv                   # Per-fold validation metrics across all 5 outer folds
│   ├── feature_importance.csv               # Ranked feature attributions
│   └── latex_results_table.tex              # Formatted booktabs LaTeX table code
├── report/
│   ├── COMPXXXX.cls                         # University of Greenwich official LaTeX class
│   ├── COMPXXXX.bst                         # University of Greenwich bibliography style
│   ├── references_AI.bib                    # Academic BibTeX citations (2024–2026)
│   ├── karkar_kirtan_AI_Intro.tex           # Complete 4-page academic report (student named)
│   ├── surname_firstname_AI_Intro.tex       # Complete 4-page report (spec named copy)
│   └── (all figures copied here)            # Self-contained for immediate Overleaf compilation
├── src/
│   ├── __init__.py                          # Package initialization
│   ├── data_loader.py                       # Data ingestion, target mapping, leak-free pipelines
│   ├── models.py                            # Trivial baselines, symbolic rule engines, and ML models
│   ├── evaluate.py                          # Nested 5x5 CV engine, test set evaluator, LaTeX builder
│   ├── explainability.py                    # SHAP TreeExplainer, permutation, and cost matrix
│   └── plotting.py                          # Publication-grade high-resolution plotting
├── train_and_evaluate.py                    # Master one-click end-to-end execution script
└── README.md                                # Project documentation and reproduction guide
```

---

### Quantitative Benchmark Results

Evaluated across $N_{\text{train}}=491$ (Nested Stratified 5-Fold Cross-Validation) and $N_{\text{test}}=123$ (Independent Held-Out Test Set):

| Model Architecture | Test Accuracy | Test Macro F1 | Test ROC-AUC | Reject Recall (0) | Approve Recall (1) | Nested CV Macro F1 | Nested CV Accuracy |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Majority Baseline** | 0.691 | 0.409 | 0.500 | 0.000 | 1.000 | 0.407 ($\pm$0.002) | 0.686 ($\pm$0.004) |
| **Random Chance Baseline** | 0.553 | 0.480 | 0.480 | 0.289 | 0.671 | 0.505 ($\pm$0.067) | 0.576 ($\pm$0.059) |
| **Credit History 1-Rule (Symbolic)** | 0.854 | 0.802 | 0.770 | 0.553 | 0.988 | 0.707 ($\pm$0.089) | 0.798 ($\pm$0.051) |
| **Symbolic Expert Rules (LO3)** | 0.854 | 0.802 | 0.770 | 0.553 | 0.988 | 0.707 ($\pm$0.089) | 0.798 ($\pm$0.051) |
| **Logistic Regression** | **0.862** | **0.815** | **0.849** | 0.579 | 0.988 | **0.711 ($\pm$0.089)** | **0.800 ($\pm$0.051)** |
| **Decision Tree (CART)** | 0.854 | 0.802 | 0.757 | 0.553 | 0.988 | 0.693 ($\pm$0.081) | 0.786 ($\pm$0.043) |
| **Random Forest** | 0.837 | 0.784 | 0.795 | 0.553 | 0.965 | 0.668 ($\pm$0.085) | 0.747 ($\pm$0.074) |
| **Support Vector Machine (SVM)** | 0.846 | 0.788 | 0.826 | 0.526 | 0.988 | 0.681 ($\pm$0.080) | 0.764 ($\pm$0.055) |
| **XGBoost Classifier** | 0.854 | 0.802 | 0.795 | 0.553 | 0.988 | 0.697 ($\pm$0.100) | 0.788 ($\pm$0.058) |
| **RF + XGBoost Soft Voting** | **0.837** | **0.784** | **0.780** | 0.553 | 0.965 | **0.692 ($\pm$0.083)** | **0.784 ($\pm$0.049)** |

---

### Step-by-Step Reproduction Instructions

#### 1. One-Click Master Execution
To replicate all experimental runs, execute nested cross-validation, calculate SHAP attributions, generate all publication figures, and export all LaTeX tables:
```bash
python3 train_and_evaluate.py
```
*Expected runtime: ~20–30 seconds.*

#### 2. Interactive Jupyter Notebook
An interactive walkthrough containing code cells, diagnostics, and narrative explanations:
```bash
jupyter notebook notebooks/comp1818_loan_approval_experiment.ipynb
```

#### 3. Uploading to Overleaf
The `report/` folder contains every dependency required for seamless LaTeX compilation:
1. Compress the `report/` folder into a `.zip` archive:
   ```bash
   cd report && zip -r ../loan_approval_report_overleaf.zip . && cd ..
   ```
2. Log into [Overleaf](https://www.overleaf.com).
3. Select **New Project** $\to$ **Upload Project**, and drag `loan_approval_report_overleaf.zip`.
4. Set main document to `karkar_kirtan_AI_Intro.tex` (or `surname_firstname_AI_Intro.tex`) and compile with standard **pdfLaTeX**.
5. It compiles cleanly in two columns with zero errors, exactly 4 pages plus references and the mandatory 60–100 word AI declaration.
