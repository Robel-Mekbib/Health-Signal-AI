# HealthSignal AI — One-Page Writeup

**Problem.** Community health workers lack tools to spot subtle diabetes risk. Early awareness saves lives, but a paper checklist is slow and imprecise.

**Solution.** A web app that takes eight health-related measurements represented in the Pima dataset, returns an estimated model score (0–100, not a clinical probability), a plain-language SHAP explanation, and a clear screening flag. It is designed as a **risk-screening awareness prototype**, not a diagnostic tool.

**AI core.** A Logistic Regression model trained on the Pima Indians Diabetes dataset (768 samples), compared against a majority-class baseline and Random Forest. We explain every prediction individually using SHAP.

**Original contributions.**
- **Rigorous validation:** An initial version of this project tuned the decision threshold directly on the test set, which artificially inflated the reported performance. We corrected this by using a strict 60/20/20 train/validation/test split: the model and threshold are now selected using only the validation set, and the test set is used exactly once.
- **Threshold tuning:** We selected 0.40 as a deliberate precision-recall trade-off. Compared with 0.25, it improves precision and reduces false positives, but lowers recall and increases false negatives. This is a prototype design choice, not a clinically validated threshold.
- **Per-prediction SHAP:** Health workers can see which input features most influenced the model's output. SHAP explains model behavior, not medical causes.
- **Honest reporting:** We report False Negatives (FN) and False Positives (FP) separately, report the Brier score as an additional measure of prediction error (without claiming it establishes clinical calibration), and perform an age-group analysis.

**Results (final held-out test set, threshold=0.40).**
- Baseline: 64.9% acc, recall 0%.
- **Logistic Regression (Deployed): 72.1% acc, 0.593 recall, 0.598 F1, 22 FN / 21 FP, AUC 0.817, Brier 0.175.**

**Limitations.** The dataset contains **adult women of Pima Indigenous descent (1990s)**, not a representative global population. It is not validated for men, children, or other ethnicities. It has a ~41% false-negative rate. The output is a model score, not a clinical diagnostic. **This project is a prototype and is not medical advice.**
