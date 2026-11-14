# HealthSignal AI — One-Page Writeup

**Problem.** Community health workers lack tools to spot subtle diabetes risk.

**Solution.** A web app that takes 8 routine measurements, returns an estimated risk probability, a plain-language SHAP explanation, and a clear screening flag.

**AI core.** Random Forest trained on Pima Indians Diabetes (768 samples), compared against baseline and Logistic Regression, explained per-prediction with SHAP.

**Original contributions.**
- Threshold tuned to 0.30 to prioritize recall (screening context).
- Per-prediction SHAP — health workers see *why*.
- Separate FN/FP reporting with honest limitations.

**Results (test set, n=154).**
- Baseline: 64.9% acc, recall 0%.
- Random Forest @ thr=0.30: **70.8% acc, 0.778 recall, 0.651 F1, 12 FN / 33 FP.**
- Top SHAP feature: Glucose.

**Limitations.** Narrow population, ~22% FN rate. **Prototype, not medical advice.**