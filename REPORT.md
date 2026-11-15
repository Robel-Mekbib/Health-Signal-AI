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

## Results (Final held-out test set, threshold=0.40)
- Baseline recall: 0% (FN=54).
- **Random Forest (Deployed): Recall 0.722, F1 0.678, FN=15, FP=22.**

**Model Selection:** While Logistic Regression is often favored for interpretability, our unbiased validation phase showed Random Forest achieved a higher F1 score (0.702 vs 0.688). Therefore, we deployed Random Forest.

## SHAP Feature Importance
*(Feature contributions describe model behavior, not medical causes.)*
Glucose is the most heavily weighted feature, followed by BMI and Age. This aligns with established clinical risk factors, confirming the model learned sensible patterns.

## Weaknesses (honestly reported)
- **≈28% false-negative rate** at thr=0.40 — acceptable only as a screening aid.
- **Insulin** is ~49% missing → its SHAP contribution is unreliable.
- **Population narrowness** — cannot generalise to men, children, other ethnicities.
- **Data age** — 1990s diagnostic criteria may differ from current standards.

## Recommendation
Deploy only as a **screening-awareness tool** for community health workers, with **mandatory professional referral** on any positive flag. Not for clinical decision-making.
