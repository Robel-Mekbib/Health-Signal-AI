import gradio as gr
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = joblib.load("healthsignal_final.pkl")
IMPUTER = MODEL.named_steps["imputer"]
SCALER = MODEL.named_steps["scaler"]
CLF = MODEL.named_steps["clf"]
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
THRESHOLD = 0.40

# LinearExplainer with a background sample for the scaled features
_bg = pd.DataFrame([{
    "Pregnancies":3,"Glucose":120,"BloodPressure":70,"SkinThickness":23,
    "Insulin":80,"BMI":32,"DiabetesPedigreeFunction":0.5,"Age":33
}])[FEATURES]
_bg_imp = IMPUTER.transform(_bg)
_bg_scaled = SCALER.transform(_bg_imp)
EXPLAINER = shap.LinearExplainer(CLF, _bg_scaled)

def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, pedigree, age):
    if any(v is None for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]):
        return "⚠️ Please fill in all fields.", None, ""
    if any(v < 0 for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]):
        return "⚠️ Negative values are not allowed.", None, ""
    if glucose < 40 or glucose > 400:
        return "⚠️ Glucose must be between 40–400 mg/dL.", None, ""
    if bmi < 10 or bmi > 70:
        return "⚠️ BMI must be between 10–70.", None, ""
    if age < 1 or age > 120:
        return "⚠️ Age must be between 1–120.", None, ""
    if blood_pressure < 30 or blood_pressure > 200:
        return "⚠️ Blood pressure must be between 30–200 mmHg.", None, ""
    if pregnancies > 20:
        return "⚠️ Pregnancies must be between 0–20.", None, ""
    if skin_thickness > 100:
        return "⚠️ Skin thickness must be ≤ 100 mm.", None, ""
    if insulin > 1000:
        return "⚠️ Insulin must be ≤ 1000 μU/mL.", None, ""
    if pedigree > 3:
        return "⚠️ Diabetes Pedigree must be ≤ 3.", None, ""
    if age < 21:
        return ("⚠️ This prototype was trained on women aged ≥21. "
                "Predictions for younger ages are outside the model's supported population.", None, "")

    row = pd.DataFrame([{
        "Pregnancies": pregnancies,
        "Glucose": glucose if glucose > 0 else np.nan,
        "BloodPressure": blood_pressure if blood_pressure > 0 else np.nan,
        "SkinThickness": skin_thickness if skin_thickness > 0 else np.nan,
        "Insulin": insulin if insulin > 0 else np.nan,
        "BMI": bmi if bmi > 0 else np.nan,
        "DiabetesPedigreeFunction": pedigree,
        "Age": age,
    }])[FEATURES]

    proba = MODEL.predict_proba(row)[0, 1]
    flagged = proba >= THRESHOLD

    if proba < 0.20:
        band, color = "LOW estimated model score", "#16a34a"
    elif proba < THRESHOLD:
        band, color = "MODERATE estimated model score", "#d97706"
    else:
        band, color = "ELEVATED estimated model score — recommend professional assessment", "#dc2626"

    result_md = f"""
<div style="padding:18px;border-radius:12px;background:#f8fafc;
            border-left:6px solid {color};">
  <div style="font-size:13px;color:#64748b;text-transform:uppercase;
              letter-spacing:1px;">Estimated model score</div>
  <div style="font-size:28px;font-weight:600;color:{color};
              margin:6px 0;">{proba*100:.1f}</div>
  <div style="font-size:15px;color:#0f172a;">{band}</div>
  <div style="font-size:12px;color:#334155;margin-top:10px;">
    Decision threshold: {THRESHOLD:.2f}. Screening flag:
    <b>{'YES' if flagged else 'NO'}</b>.
  </div>
  <div style="font-size:11px;color:#64748b;margin-top:8px;">
    This is a model score (0–100), not a clinical risk probability.
    It is not validated for diagnostic use.
  </div>
</div>
"""

    row_imp = IMPUTER.transform(row)
    row_scaled = SCALER.transform(row_imp)
    sv = EXPLAINER.shap_values(row_scaled)[0]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    order = np.argsort(np.abs(sv))
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in sv[order]]
    ax.barh(np.array(FEATURES)[order], sv[order], color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("Contribution to model score (SHAP value)")
    ax.set_title("Why this score? (red = raises, blue = lowers). "
                 "Feature contributions describe model behavior, not medical causes.",
                 fontsize=9, loc="left")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    top = pd.Series(np.abs(sv), index=FEATURES).sort_values(ascending=False)
    top3 = top.head(3)
    explain_lines = [
        f"**{feat}** — measured **{row.iloc[0][feat] if not pd.isna(row.iloc[0][feat]) else 'not provided'}**"
        + (" (raises score)" if sv[FEATURES.index(feat)] > 0 else " (lowers score)")
        for feat in top3.index
    ]
    explanation_md = ("### Top contributing factors\n\n"
                      + "\n".join(f"- {l}" for l in explain_lines)
                      + "\n\n*These contributions describe how the model behaves. "
                        "They are not medical causes.*")
    return result_md, fig, explanation_md

with gr.Blocks(title="HealthSignal AI") as demo:
    gr.Markdown("""
    # HealthSignal AI
    ### Intelligent Insights for Earlier Health Awareness

    Community health workers: enter routine measurements to estimate diabetes
    model score and identify individuals who may benefit from professional assessment.

    > ⚠️ **This project is a prototype and is not medical advice.**
    > It does not diagnose diabetes and does not replace a clinician.
    """)
    gr.Markdown("""
    > ⚠️ **Trained on women of Pima Indigenous descent, aged ≥21, 1990s data.**
    > **Not validated for males, children, or any other population group.**
    > This tool is a **screening-awareness prototype for community health workers** —
    > it does not replace professional assessment and is not a diagnostic device.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Health measurements")
            pregnancies = gr.Number(label="Pregnancies", value=2, precision=0)
            glucose     = gr.Number(label="Glucose (mg/dL)", value=120)
            bp          = gr.Number(label="Blood Pressure (mmHg)", value=70)
            skin        = gr.Number(label="Skin Thickness (mm, 0 = unknown)", value=0)
            insulin     = gr.Number(label="Insulin (μU/mL, 0 = unknown)", value=0)
            bmi         = gr.Number(label="BMI", value=28.5)
            pedigree    = gr.Number(label="Diabetes Pedigree Function", value=0.5)
            age         = gr.Number(label="Age (years)", value=35)
            btn = gr.Button("Estimate model score", variant="primary", size="lg")

        with gr.Column(scale=2):
            out_result = gr.HTML()
            out_expl   = gr.Markdown()
            out_plot   = gr.Plot()

    btn.click(predict,
              inputs=[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age],
              outputs=[out_result, out_plot, out_expl])

    with gr.Accordion("About this model & its limitations", open=False):
        gr.Markdown("""
        **Model:** Logistic Regression (interpretable, linear), trained on the Pima Indians Diabetes dataset (768 samples).
        **Decision threshold:** 0.40 — selected on a validation set, not the test set.
        **Why Logistic Regression?** Its coefficients are directly interpretable, and it performed comparably to Random Forest on the unbiased validation set. Choosing the simpler, interpretable model is appropriate for a screening aid used by non-specialists.

        **Limitations you must know:**
        - Trained on **women of Pima Indigenous descent** aged ≥21. Not validated for males, children, or other ethnic groups.
        - Data collected in the **1990s**; diagnostic criteria may have shifted.
        - **Insulin and Skin Thickness are ~30–49% missing** — SHAP contributions from these are unreliable.
        - The model score is **not a calibrated clinical probability**.
        - **This is a screening-awareness prototype, not a diagnostic tool.**

        **This project is a prototype and is not medical advice.**
        """)

demo.launch(share=False, debug=True)
