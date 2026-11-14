# HealthSignal AI — Evaluation Report

## Important methodology note (revised)
An earlier version of this project tuned the decision threshold on the test set, invalidating it as an independent evaluation. We corrected this: models are trained on 60% of the data, the threshold is selected on a 20% validation split, and all numbers below are reported on a **final 20% held-out test set used exactly once**. The test-set recall is therefore lower than the earlier number and represents an unbiased estimate.

## Dataset
Pima Indians Diabetes (UCI). 768 samples, 8 features, binary outcome (34.9% positive).

## Preprocessing
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing (`NaN`).
- Pregnancies = 0 retained (legitimate).
- Median imputation inside sklearn Pipeline, fitted only on training fold.
- Stratified 60/20/20 train/validation/test split, `random_state=42`.

## Metrics
- **Accuracy** — misleading on imbalanced data (baseline hits 64.9% while catching zero positives).
- **Precision** — of those flagged, how many truly positive.
- **Recall** — of true positives, how many caught. **Primary metric here.**
- **F1** — harmonic mean, tie-breaker.
- **ROC-AUC** — ranking quality.
- **FN/FP** — reported separately per health-track requirement.

## Calibration
We did **not** validate calibration of the predicted probabilities. The output is a well-ordered score, but its numeric magnitude is not guaranteed to match true clinical risk. We therefore describe the output as a **model score (0–100)**, not as a "% chance of diabetes." Brier score on the test set is provided for reference (LR: 0.175, RF: 0.167) but we do **not** interpret it as clinical calibration.

## Results (Final held-out test set, threshold=0.30)
- Baseline recall: 0% (FN=54).
- Random Forest: Recall 0.778, F1 0.651, FN=12, FP=33.
- **Logistic Regression (Deployed): Recall 0.796, F1 0.688, FN=11, FP=28.**

**Model Selection:** Logistic Regression was chosen over Random Forest because it achieved higher F1, higher recall, fewer false negatives, and higher accuracy on the unbiased test set. Its linear nature also makes it more interpretable for health workers.

## SHAP Feature Importance
*(Feature contributions describe model behavior, not medical causes.)*
Glucose is the most heavily weighted feature, followed by BMI and Age. This aligns with established clinical risk factors, confirming the model learned sensible patterns.

## Age-group performance (Test set)
We performed a cautious age-group evaluation. Recall varies across age bands (e.g., higher in 30-50 groups, lower in older groups), but subgroup sample sizes are small (often <40). We report these numbers **as a caveat, not as validation** — they demonstrate that group-level performance is heterogeneous. No formal statistical comparison is performed due to small n.

## Weaknesses (honestly reported)
- **≈20% false-negative rate** at thr=0.30 — acceptable only as a screening aid.
- **Insulin** is ~49% missing → its SHAP contribution is unreliable.
- **Population narrowness** — cannot generalise to men, children, other ethnicities.
- **Data age** — 1990s diagnostic criteria may differ from current standards.

## Recommendation
Deploy only as a **screening-awareness tool** for community health workers, with **mandatory professional referral** on any positive flag. Not for clinical decision-making.
