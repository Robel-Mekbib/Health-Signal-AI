# HealthSignal AI
**Intelligent Insights for Earlier Health Awareness**

An AI-powered diabetes risk-screening prototype for community health workers.
Built for the **Veritas AI International High School Hackathon 2026** — AI + Health track.

> ⚠️ **This project is a prototype and is not medical advice.**
> It does not diagnose diabetes and does not replace a clinician.

## What it does
Community health workers enter routine measurements (glucose, BMI, blood pressure, age, etc.). The model returns an **estimated model score (0–100)** — *not a clinical probability* — an **explanation of which features drove the prediction** (SHAP), and **clear guidance to seek professional assessment** when the score is elevated.

## How it works
- **Dataset:** Pima Indians Diabetes (UCI / Kaggle, 768 samples, 8 features).
- **Preprocessing:** Physiologically impossible zeros treated as missing; median imputation and scaling inside a sklearn Pipeline to prevent leakage.
- **Models compared:** Majority-class baseline, Logistic Regression, Random Forest.
- **Chosen model:** **Logistic Regression** (selected for interpretability and clinical transparency).
- **Explainability:** SHAP LinearExplainer, per-prediction.
- **Decision threshold:** 0.40 — tuned on a validation set (not the test set) to balance recall and precision.

## 🛡️ Methodology Correction (Transparency)
An earlier version of this project tuned the decision threshold on the test set, invalidating it as an independent evaluation. **We corrected this.** Models are trained on 60% of the data, the threshold is selected on a 20% validation split, and all numbers below are reported on a **final 20% held-out test set used exactly once**. The results represent an unbiased estimate.

## Final Evaluation (Unbiased Test Set, Logistic Regression, threshold=0.40)

| Model | Acc | Prec | Rec | F1 | AUC | Brier | FN | FP |
|---|---|---|---|---|---|---|---|---|
| Baseline (majority) | 0.649 | 0.000 | 0.000 | 0.000 | 0.500 | 0.351 | 54 | 0 |
| **Logistic Regression (Deployed)** | **0.721** | **0.604** | **0.593** | **0.598** | **0.817** | **0.175** | **22** | **21** |

**Model Selection:** On the validation set, Random Forest achieved a slightly higher F1 (0.702 vs 0.673). However, we deployed Logistic Regression because its coefficients map directly to feature influence, making it fully interpretable for non-specialist health workers. This transparency is critical for a screening aid.

### Age-Group Performance (Test Set)
- **Age 21–30:** n=85, Recall=0.438
- **Age 30–40:** n=30, Recall=0.600
- **Age 40–50:** n=24, Recall=0.833
- **Age 50–100:** n=15, Recall=0.500
*Subgroup sample sizes are small (often <40). We report these as a caveat, not as validation.*

## Limitations
- Trained only on **women of Pima Indigenous descent aged ≥21**. Not validated for males, children, or other populations.
- Data collected in the **1990s**.
- **Insulin and Skin Thickness are ~30–49% missing** — SHAP contributions from these are unreliable.
- At threshold 0.40 the model misses **22 of 54 positive cases** (≈41% false negative rate).
- The Brier score (0.175) indicates moderate calibration; the output is a **model score**, not a clinical probability.
- **This project is a prototype and is not medical advice.**

## Files
- `app.py` — Gradio web interface
- `healthsignal_final.pkl` — trained Logistic Regression model
- `HealthSignal_ai.ipynb` — full training and evaluation notebook
- `REPORT.md` — detailed evaluation report
- `writeup.md` — one-page summary

## Citations
- Dataset: Smith et al. (1988), *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
- Dataset Source URL: `https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv`
- SHAP: Lundberg & Lee (2017), NeurIPS.
- Libraries: scikit-learn, pandas, NumPy, Gradio, matplotlib.
- **AI Tools:** Claude (Anthropic) for architectural guidance, code review, and documentation drafting.

## Disclaimer
**This project is a prototype and is not medical advice.**
