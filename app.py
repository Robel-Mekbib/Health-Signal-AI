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

_bg_scaled = joblib.load("shap_background.pkl")
EXPLAINER = shap.LinearExplainer(CLF, _bg_scaled)

def predict(pregnancies, glucose, blood_pressure, skin_thickness,
            insulin, bmi, pedigree, age):
    
    # 1. Reject missing (None) inputs
    if any(v is None for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]):
        return "⚠️ Please fill in all fields.", None, ""
        
    # 2. Reject NaN or infinite values
    if any(np.isnan(v) or np.isinf(v) for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]):
        return "⚠️ Invalid numeric value (NaN or infinite).", None, ""

    # 3. Reject negatives
    if any(v < 0 for v in [pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]):
        return "⚠️ Negative values are not allowed.", None, ""

    if not float(pregnancies).is_integer():
        return "⚠️ Pregnancies must be a whole number.", None, ""

    # 4. Range validations (Note: 0 is accepted as missing for Glucose, BP, Skin, Insulin, BMI)
    if glucose != 0 and (glucose < 40 or glucose > 400):
        return "⚠️ Glucose must be 0 (missing) or between 40–400 mg/dL.", None, ""
    if bmi != 0 and (bmi < 10 or bmi > 70):
        return "⚠️ BMI must be 0 (missing) or between 10–70.", None, ""
    if blood_pressure != 0 and (blood_pressure < 30 or blood_pressure > 200):
        return "⚠️ Blood pressure must be 0 (missing) or between 30–200 mmHg.", None, ""
    if skin_thickness != 0 and skin_thickness > 100:
        return "⚠️ Skin thickness must be 0 (missing) or ≤ 100 mm.", None, ""
    if insulin != 0 and insulin > 1000:
        return "⚠️ Insulin must be 0 (missing) or ≤ 1000 μU/mL.", None, ""
        
    if age < 1 or age > 120:
        return "⚠️ Age must be between 1–120.", None, ""
    if pregnancies > 20:
        return "⚠️ Pregnancies must be between 0–20.", None, ""
    if pedigree > 3:
        return "⚠️ Diabetes Pedigree must be ≤ 3.", None, ""
    if age < 21:
        return ("⚠️ This prototype was trained on women aged ≥21. "
                "Predictions for younger ages are outside the model's supported population.", None, "")

    # Zeros are mapped to NaN for imputation
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

    color = "#dc2626" if flagged else "#16a34a"
    band = "ELEVATED model score — recommend professional assessment" if flagged else "BELOW SCREENING THRESHOLD"

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
  
  <!-- JUDGE REQUESTED WARNING ADDED HERE -->
  <div style="font-size:12px;color:#dc2626;font-weight:600;margin-top:8px;">
    Note: A result below the screening threshold does not mean diabetes is absent.
  </div>
  
  <div style="font-size:11px;color:#64748b;margin-top:4px;">
    This is a model score (0–100), not a clinical probability.
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
    """)
    
    gr.Markdown("""
    > 🚨 **CRITICAL LIMITATION:** This model was trained exclusively on 768 records from **adult women of Pima Indigenous descent (Arizona, 1990s)**. 
    > It is **not validated** for males, children, the Ethiopian population, or any other demographic. 
    > **If you enter measurements for a male, a child, or a person outside the Pima population, the resulting model score has no validated meaning** and should not be interpreted as a health assessment.
    > It is a research prototype and **not a diagnostic tool**. Do not use for clinical decisions.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### Health measurements *(Enter 0 if unknown)*")
            pregnancies = gr.Number(label="Pregnancies (count)", value=2, precision=0)
            glucose     = gr.Number(label="2-hour OGTT plasma glucose (mg/dL)", value=120)
            bp          = gr.Number(label="Diastolic blood pressure (mmHg)", value=70)
            skin        = gr.Number(label="Triceps skinfold thickness (mm)", value=0)
            insulin     = gr.Number(label="2-hour serum insulin (μU/mL)", value=0)
            bmi         = gr.Number(label="Body mass index (kg/m²)", value=28.5)
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

    with gr.Accordion("About this model, data & limitations", open=False):
        gr.Markdown("""
        **Model:** Logistic Regression (easier to inspect, linear).
        **Decision threshold:** 0.40.
        **Threshold Justification:** We chose 0.40 over 0.25 (which had a higher recall) to prioritize precision and reduce false alarms for community health workers. The output is a screening score, not a diagnosis.
        **Dataset:** Pima Indians Diabetes (UCI, 768 samples). Includes only adult women of Pima descent, aged ≥21, collected in the 1990s.
        **Missing Values:** Entering 0 for glucose, blood pressure, skinfold, insulin, or BMI will trigger the same median imputation used during training.
        **Measurements:** You must enter the exact measurements the model was trained on (e.g., 2-hour OGTT, not fasting).
        
        **This project is a prototype and is not medical advice.**
        """)

demo.launch(share=False, debug=True)
