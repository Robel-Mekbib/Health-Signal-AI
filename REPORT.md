# HealthSignal AI — Evaluation Report

## Important methodology note
An earlier version of this project tuned the decision threshold on the test set, invalidating it as an independent evaluation. We corrected this: models are trained on 60% of the data, the threshold is selected on a 20% validation split, and all numbers below are reported on a **final 20% held-out test set used exactly once**.

## Dataset
Pima Indians Diabetes (UCI). 768 samples, 8 features, binary outcome (34.9% positive).

## Preprocessing
- Zeros in Glucose, BloodPressure, SkinThickness, Insulin, BMI treated as missing (`NaN`).
- Pregnancies = 0 retained (legitimate).
- Median imputation and scaling inside sklearn Pipeline, fitted only on training fold.
- Stratified 60/20/20 train/validation/test split, `random_state=42`.

## Metrics
- **Accuracy** — misleading on imbalanced data.
- **Precision** — of those flagged, how many truly positive.
- **Recall** — of true positives, how many caught. **Primary metric.**
- **F1** — harmonic mean.
- **ROC-AUC** — ranking quality.
- **Brier** — probability calibration (lower is better).
- **FN/FP** — reported separately per health-track requirement.

## Results (Final held-out test set, Logistic Regression, threshold=0.40)
- **Logistic Regression (Deployed):** Accuracy 0.721, Precision 0.604, Recall 0.593, F1 0.598, AUC 0.817, Brier 0.175, FN=22, FP=21.

**Model Selection:** Random Forest achieved a higher F1 on the validation set (0.702 vs 0.673). We deployed Logistic Regression for its superior interpretability, accepting a minor F1 drop to ensure clinical transparency.

## Age-Group Performance (Test Set)
- **21–30:** n=85, Recall=0.438
- **30–40:** n=30, Recall=0.600
- **40–50:** n=24, Recall=0.833
- **50–100:** n=15, Recall=0.500
*Subgroup sample sizes are small. We report these as a caveat, not as validation.*

## SHAP Feature Importance
*(Feature contributions describe model behavior, not medical causes.)*
Glucose is the most heavily weighted feature, followed by BMI and Age.

## Weaknesses
- **≈41% false-negative rate** at thr=0.40. A screening aid only.
- **Insulin** is ~49% missing → SHAP contribution unreliable.
- **Population narrowness** — cannot generalise to men, children, other ethnicities.
- **Data age** — 1990s criteria may differ.
- **Calibration** — Brier 0.175 indicates moderate calibration; not a clinical probability.

## Recommendation
Deploy only as a **screening-awareness tool** with **mandatory professional referral** on any positive flag. Not for clinical decision-making.
