# HealthSignal AI — One-Page Writeup

**Problem.** Community health workers lack tools to spot subtle diabetes risk. Early awareness saves lives, but a paper checklist is slow and imprecise.

**Solution.** A web app that takes 8 routine measurements, returns an estimated model score (0–100, not a clinical probability), a plain-language SHAP explanation, and a clear screening flag.

**AI core.** A Logistic Regression model trained on the Pima Indians Diabetes dataset (768 samples), compared against a majority-class baseline and Random Forest. We explain every prediction individually using SHAP.

**Original contributions.**
- **Rigorous validation:** We detected and corrected a data leakage issue from our initial threshold tuning. We rebuilt the pipeline with a strict train/validation/test split (60/20/20).
- **Threshold tuning:** We tuned the decision threshold to 0.40 on the validation set to balance recall and precision.
- **Per-prediction SHAP:** Health workers see *why* a score was generated.
- **Honest reporting:** We report False Negatives (FN) and False Positives (FP) separately, report Brier score for calibration, and explicitly state population limitations.

**Results (final held-out test set, threshold=0.40).**
- Baseline: 64.9% acc, recall 0%.
- **Logistic Regression (Deployed): 72.1% acc, 0.593 recall, 0.598 F1, 22 FN / 21 FP, AUC 0.817, Brier 0.175.**
- Top SHAP feature: Glucose (aligns with clinical literature).

**Limitations.** Trained only on women of Pima Indigenous descent aged ≥21 in the 1990s. ~41% false-negative rate. The output is a model score, not a clinical diagnostic. **This project is a prototype and is not medical advice.**
