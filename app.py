import gradio as gr
import joblib
import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MODEL = joblib.load("healthsignal_rf.pkl")
EXPLAINER = shap.TreeExplainer(MODEL.named_steps["clf"])
FEATURES = ["Pregnancies","Glucose","BloodPressure","SkinThickness",
            "Insulin","BMI","DiabetesPedigreeFunction","Age"]
THRESHOLD = 0.30

def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, pedigree, age):
    if glucose < 40 or glucose > 400:
        return "⚠️ Glucose must be between 40–400 mg/dL.", None, ""
    if bmi < 10 or bmi > 70:
        return "⚠️ BMI must be between 10–70.", None, ""
    if age < 1 or age > 120:
        return "⚠️ Age must be between 1–120.", None, ""
    if blood_pressure < 30 or blood_pressure > 200:
        return "⚠️ Blood pressure must be between 30–200 mmHg.", None, ""
    if pregnancies < 0 or pregnancies > 20:
        return "⚠️ Pregnancies must be between 0–20.", None, ""

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
        band, color = "LOW estimated risk", "#16a34a"
    elif proba < THRESHOLD:
        band, color = "MODERATE estimated risk", "#d97706"
    else:
        band, color = "ELEVATED estimated risk — recommend professional assessment", "#dc2626"

    result_md = f"""
<div style="padding:18px;border-radius:12px;background:#f8fafc;
            border-left:6px solid {color};">
  <div style="font-size:13px;color:#64748b;text-transform:uppercase;
              letter-spacing:1px;">Estimated risk</div>
  <div style="font-size:28px;font-weight:600;color:{color};
              margin:6px 0;">{proba*100:.1f}%</div>
  <div style="font-size:15px;color:#0f172a;">{band}</div>
  <div style="font-size:12px;color:#334155;margin-top:10px;">
    Decision threshold: {THRESHOLD:.2f}. Screening flag:
    <b>{'YES' if flagged else 'NO'}</b>.
  </div>
</div>
"""

    row_imp = MODEL.named_steps["imputer"].transform(row)
    sv = EXPLAINER.shap_values(row_imp)
    if isinstance(sv, list):
        sv = sv[1]
    elif sv.ndim == 3:
        sv = sv[:, :, 1]
    sv = sv[0]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    order = np.argsort(np.abs(sv))
    colors = ["#dc2626" if v > 0 else "#2563eb" for v in sv[order]]
    ax.barh(np.array(FEATURES)[order], sv[order], color=colors)
    ax.axvline(0, color="#94a3b8", linewidth=0.8)
    ax.set_xlabel("Contribution to risk (SHAP value)")
    ax.set_title("Why this estimate? (red = raises risk, blue = lowers)",
                 fontsize=10, loc="left")
    ax.spines[["top","right"]].set_visible(False)
    plt.tight_layout()

    top = pd.Series(np.abs(sv), index=FEATURES).sort_values(ascending=False)
    top3 = top.head(3)
    explain_lines = [
        f"**{feat}** — measured **{row.iloc[0][feat] if not pd.isna(row.iloc[0][feat]) else 'not provided'}**"
        + (" (raises risk)" if sv[FEATURES.index(feat)] > 0 else " (lowers risk)")
        for feat in top3.index
    ]
    explanation_md = "### Top contributing factors\n\n" + "\n".join(f"- {l}" for l in explain_lines)

    return result_md, fig, explanation_md


with gr.Blocks(title="HealthSignal AI") as demo:
    gr.Markdown("""
    # HealthSignal AI
    ### Intelligent Insights for Earlier Health Awareness

    Community health workers: enter routine measurements to estimate diabetes risk
    and identify individuals who may benefit from professional assessment.

    > ⚠️ **This project is a prototype and is not medical advice.**
    > It does not diagnose diabetes and does not replace a clinician.
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
            btn = gr.Button("Estimate risk", variant="primary", size="lg")

        with gr.Column(scale=2):
            out_result = gr.HTML()
            out_expl   = gr.Markdown()
            out_plot   = gr.Plot()

    btn.click(predict,
              inputs=[pregnancies, glucose, bp, skin, insulin, bmi, pedigree, age],
              outputs=[out_result, out_plot, out_expl])

    with gr.Accordion("About this model & its limitations", open=False):
        gr.Markdown("""
        **Model:** Random Forest (300 trees), trained on the Pima Indians Diabetes dataset (768 samples).
        **Features:** Pregnancies, Glucose, Blood Pressure, Skin Thickness, Insulin, BMI, Diabetes Pedigree Function, Age.
        **Decision threshold:** 0.30 — chosen to prioritize recall (fewer missed cases) over precision, appropriate for a screening tool.

        **Limitations you must know:**
        - Trained on **women of Pima Indigenous descent** aged ≥21. Not validated for males, children, or other ethnic groups.
        - Data collected in the **1990s**; diagnostic criteria may have shifted.
        - **Insulin and Skin Thickness are ~30–49% missing** in the source data — treat those contributions with caution.
        - On the held-out test set the model missed **12 of 54 positive cases** (≈22% false negative rate). It is a screening aid, not a diagnosis.
        - Performance differences across demographic subgroups were **not** evaluated — the dataset lacks that granularity.

        **This project is a prototype and is not medical advice.**
        """)

demo.launch(theme=gr.themes.Soft(), share=False, debug=False)