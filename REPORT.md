# HealthSignal AI — Evaluation Report

## 1. Threshold Selection & Justification
We evaluated multiple thresholds on the validation set. 
- **Threshold 0.25:** Recall = 0.815, Precision = 0.595, F1 = 0.688, FP = 30, FN = 10.
- **Threshold 0.40:** Recall = 0.667, Precision = 0.679, F1 = 0.673, FP = 17, FN = 18.

We selected **0.40** as a deliberate precision-recall trade-off. Compared to 0.25, threshold 0.40 improves validation precision and reduces false positives (from 30 to 17), but it also lowers recall and increases false negatives (from 10 to 18). This is a prototype design choice favoring fewer false alarms over catching every possible case; it is **not** a clinically validated threshold.

## 2. Final Evaluation (Held-out Test Set, thr=0.40)
- **Logistic Regression:** Acc=0.721, Prec=0.604, Rec=0.593, F1=0.598, AUC=0.817, Brier=0.175.
- **False Negatives:** 22 (missed positive cases)
- **False Positives:** 21 (unnecessary referrals)

## 3. Age-Group Performance (Exploratory Only)
- [21, 30): n=85, Recall=0.438
- [30, 40): n=30, Recall=0.600
- [40, 50): n=24, Recall=0.833
- [50, 100): n=15, Recall=0.500

**Warning:** Subgroup sample sizes are extremely small (often <30). These results are **exploratory and not statistically robust**. We make no claims that the model performs better or worse for specific age groups.

## 4. Calibration
The Brier score on the test set is 0.175. This is reported as an additional measure of prediction error. It does **not** establish clinical calibration. The output is a *model score*, not a probability of disease.

## 5. Limitations
- **Population:** Only adult women of Pima Indigenous descent (1990s). Not validated for men, children, or other ethnicities.
- **Measurement specificity:** Requires 2-hour OGTT and specific blood pressure/skinfold measurements.
- **Missing data:** Insulin and skinfold are ~30-49% missing in the source data.
