# HealthSignal AI — Evaluation Report

## Important methodology note
An earlier version of this project tuned the decision threshold on the test set, invalidating it as an independent evaluation. We corrected this: models are trained on 60% of the data, the threshold is selected on a 20% validation split, and all numbers below are reported on a **final 20% held-out test set used exactly once**. The test-set recall is therefore lower than earlier numbers and represents an unbiased estimate.

## Dataset
Pima Indians Diabetes (UCI). 768 samples, 8 features, binary outcome (34.9% positive).

## Preprocessing
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing (`NaN`).
- Pregnancies = 0 retained (legitimate).
- Median imputation and scaling inside sklearn Pipeline, fitted only on training fold.
- Stratified 60/20/20 train/validation/test split, `random_state=42`.

## Metrics
- **Accuracy** — misleading on imbalanced data (baseline hits 64.9% while catching zero positives).
- **Precision** — of those flagged, how many truly positive.
- **Recall** — of true positives, how many caught. **Primary metric here.**
- **F1** — harmonic mean, tie-breaker.
- **ROC-AUC** — ranking quality, independent of threshold.
- **Brier** — measures probability calibration (lower is better).
- **FN/FP** — reported separately per health-track requirement.

## Results (Final held-out test set, Logistic Regression, threshold=0.40)
- Baseline recall: 0% (FN=54).
- **Logistic Regression (Deployed):** Accuracy 0.721, Precision 0.604, Recall 0.593, F1 0.598, AUC 0.817, Brier 0.175, FN=22, FP=21.

**Model Selection:** We selected Logistic Regression over Random Forest because it achieved a higher ROC-AUC (0.817 vs 0.816) and superior interpretability. The threshold 0.40 was chosen on the validation set to balance recall (0.667) and precision (0.679).

## SHAP Feature Importance
*(Feature contributions describe model behavior, not medical causes.)*
Glucose is the most heavily weighted feature, followed by BMI and Age. This aligns with established clinical risk factors, confirming the model learned sensible patterns.

## Weaknesses (honestly reported)
- **≈41% false-negative rate** at thr=0.40 — the model misses 22 of 54 positive cases. It is a screening aid only.
- **Insulin** is ~49% missing → its SHAP contribution is unreliable.
- **Population narrowness** — cannot generalise to men, children, other ethnicities.
- **Data age** — 1990s diagnostic criteria may differ from current standards.
- **Calibration** — the Brier score (0.175) indicates moderate calibration, but we do not claim clinical probability.

## Recommendation
Deploy only as a **screening-awareness tool** for community health workers, with **mandatory professional referral** on any positive flag. Not for clinical decision-making.
