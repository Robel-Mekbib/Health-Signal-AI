# HealthSignal AI — Evaluation Report

## Dataset
Pima Indians Diabetes (UCI). 768 samples, 8 features, binary outcome (34.9% positive).

## Preprocessing
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing (NaN).
- Pregnancies = 0 retained (legitimate).
- Median imputation inside sklearn Pipeline, fitted only on training fold.
- Stratified 80/20 split, `random_state=42`.

## Metrics
- **Accuracy** — misleading on imbalanced data (baseline hits 64.9% while catching zero positives).
- **Precision** — of those flagged, how many truly positive.
- **Recall** — of true positives, how many caught. **Primary metric here.**
- **F1** — harmonic mean, tie-breaker.
- **ROC-AUC** — ranking quality.
- **FN/FP** — reported separately per health-track requirement.

## Results
- Baseline recall 0%.
- Random Forest at thr=0.30: recall 0.778, FN=12, FP=33.
- Threshold tuning is the single largest improvement.

## SHAP feature importance (mean |SHAP|)
| Feature | Mean \|SHAP\| |
|---|---|
| Glucose | 0.1346 |
| BMI | 0.0734 |
| Age | 0.0586 |
| DiabetesPedigreeFunction | 0.0385 |
| Insulin | 0.0354 |
| Pregnancies | 0.0268 |
| SkinThickness | 0.0208 |
| BloodPressure | 0.0130 |

## Weaknesses
- ≈22% false-negative rate at thr=0.30.
- Insulin ~49% missing → unreliable SHAP.
- Population narrowness.
- Data age (1990s).
- No subgroup analysis possible.