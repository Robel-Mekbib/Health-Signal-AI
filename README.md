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
- **Preprocessing:** Physiologically impossible zeros treated as missing; median imputation inside a sklearn Pipeline to prevent leakage.
- **Models compared:** Majority-class baseline, Logistic Regression, Random Forest.
- **Chosen model:** **Logistic Regression**. It outperformed Random Forest on the final held-out test set and offers greater interpretability.
- **Explainability:** SHAP LinearExplainer, per-prediction.
- **Decision threshold:** 0.30 — tuned on a validation set (not the test set) to prioritize recall for screening.

## 🛡️ Methodology Correction (Transparency)
An earlier version of this project tuned the decision threshold on the test set, invalidating it as an independent evaluation. **We corrected this.** Models are trained on 60% of the data, the threshold is selected on a 20% validation split, and all numbers below are reported on a **final 20% held-out test set used exactly once**. The results represent an unbiased estimate.

## Evaluation (Final held-out test set, n=154, threshold=0.30)
| Model | Acc | Prec | Rec | F1 | AUC | FN | FP |
|---|---|---|---|---|---|---|---|
| Baseline (majority) | 0.649 | 0.000 | 0.000 | 0.000 | 0.500 | 54 | 0 |
| Random Forest | 0.708 | 0.560 | 0.778 | 0.651 | 0.816 | 12 | 33 |
| **Logistic Regression (Deployed)** | **0.747** | **0.606** | **0.796** | **0.688** | **0.817** | **11** | **28** |

## Limitations
- Trained only on **women of Pima Indigenous descent aged ≥21**. Not validated for males, children, or other populations.
- Data collected in the **1990s**.
- **Insulin and Skin Thickness are ~30–49% missing** — SHAP contributions from these are unreliable.
- At threshold 0.30 the model still misses **11 of 54 positive cases** (≈20% false negative rate).
- The output is a **model score**, not a calibrated clinical risk.
- **This project is a prototype and is not medical advice.**

## Files
- `app.py` — Gradio web interface
- `healthsignal_logreg.pkl` — trained Logistic Regression model
- `Health_Signal_AI.ipynb` — full training and evaluation notebook
- `REPORT.md` — detailed evaluation report
- `writeup.md` — one-page summary

## Citations
- Dataset: Smith et al. (1988), *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
- SHAP: Lundberg & Lee (2017), NeurIPS.
- Libraries: scikit-learn, pandas, NumPy, Gradio, matplotlib.
- **AI Tools:** Claude (Anthropic) for architectural guidance, code review, and documentation drafting. All model design, threshold reasoning, and evaluation choices were made by the team.

## Disclaimer
**This project is a prototype and is not medical advice.**
