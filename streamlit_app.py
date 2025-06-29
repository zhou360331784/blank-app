import streamlit as st
import numpy as np
import plotly.graph_objects as go
from xhtml2pdf import pisa
from io import BytesIO
import base64
from datetime import datetime

# Configure page
st.set_page_config(page_title="NAFPD Risk Assessment", layout="centered")

# Page title and description
st.title("Non-alcoholic fatty pancreatic disease (NAFPD) Risk Assessment Tool")
st.markdown("""
Enter the clinical parameters below. The system will automatically calculate the risk probability.

---

**Notes:**
- Age > 65, FPG > 6.1 mmol/L, GGT > 50 U/L, Waist Circumference > 88.493 cm, NLR > 1.97 are flagged as high-risk factors.
- The tool computes FLI and mFIB-4 indices and applies a logistic regression model to estimate risk.
""")

# Sidebar inputs
col1, col2 = st.columns(2)
with col1:
    age = st.number_input("Age (years)", min_value=1, max_value=120, value=50)
    fpg = st.number_input("Fasting Plasma Glucose (mmol/L)", min_value=2.0, max_value=30.0, value=5.5, step=0.1)
    ggt = st.number_input("GGT (U/L)", min_value=5.0, max_value=500.0, value=30.0, step=1.0)
    waist = st.number_input("Waist Circumference (cm)", min_value=30.0, max_value=150.0, value=85.0, step=0.1)
    nlr = st.number_input("Neutrophil/Lymphocyte Ratio (NLR)", min_value=0.1, max_value=15.0, value=1.5, step=0.1)
with col2:
    triglycerides = st.number_input("Triglycerides (mg/dL)", min_value=10.0, max_value=1000.0, value=150.0, step=1.0)
    bmi = st.number_input("Body Mass Index (kg/m²)", min_value=10.0, max_value=50.0, value=23.0, step=0.1)
    ast = st.number_input("AST (U/L)", min_value=5.0, max_value=500.0, value=20.0, step=1.0)
    alt = st.number_input("ALT (U/L)", min_value=5.0, max_value=500.0, value=25.0, step=1.0)
    platelet = st.number_input("Platelet Count (10⁹/L)", min_value=10.0, max_value=1000.0, value=200.0, step=1.0)

st.markdown("---")

# 1. Calculate FLI
logit_fli = (
    0.953 * np.log(triglycerides) +
    0.139 * bmi +
    0.718 * np.log(ggt) +
    0.053 * waist -
    15.745
)
fli = np.exp(logit_fli) / (1 + np.exp(logit_fli)) * 100

# 2. Calculate mFIB-4
mfib4 = 10 * age * ast / (platelet * alt)

# 3. Binary encoding of risk factors
x_age = 1 if age > 65 else 0
x_fpg = 1 if fpg > 6.1 else 0
x_ggt = 1 if ggt > 50 else 0
x_waist = 1 if waist > 88.493 else 0
x_fli = 1 if fli > 24.7 else 0
x_mfib4 = 1 if mfib4 > 3.05 else 0
x_nlr = 1 if nlr > 1.97 else 0

# 4. Logistic regression calculation
intercept = -3.089
coefficients = {
    'Age>65': 0.748,
    'FPG>6.1': 0.903,
    'GGT>50': 0.510,
    'Waist>88.493': 0.721,
    'FLI>24.7': 0.589,
    'mFIB4>3.05': 0.731,
    'NLR>1.97': 0.458
}
logit_p = (
    intercept +
    coefficients['Age>65'] * x_age +
    coefficients['FPG>6.1'] * x_fpg +
    coefficients['GGT>50'] * x_ggt +
    coefficients['Waist>88.493'] * x_waist +
    coefficients['FLI>24.7'] * x_fli +
    coefficients['mFIB4>3.05'] * x_mfib4 +
    coefficients['NLR>1.97'] * x_nlr
)
probability = 1 / (1 + np.exp(-logit_p))

# 5. Display results
st.subheader("Intermediate Results")
st.write(f"FLI: {fli:.2f}")
st.write(f"mFIB-4: {mfib4:.2f}")

st.subheader("Risk Probability")
st.write(f"Estimated risk of NAFPD: {probability*100:.1f}%")

if probability < 0.2:
    st.success("Low risk")
elif probability < 0.5:
    st.warning("Moderate risk - consider regular monitoring")
else:
    st.error("High risk - recommend further evaluation")

st.markdown("---")

# 6. Radar chart of risk factors
st.subheader("Risk Factor Radar Chart")
factors = ['Age>65', 'FPG>6.1', 'GGT>50', 'Waist>88.493', 'FLI>24.7', 'mFIB4>3.05', 'NLR>1.97']
values = [x_age, x_fpg, x_ggt, x_waist, x_fli, x_mfib4, x_nlr]
radar_fig = go.Figure(data=go.Scatterpolar(
    r=values + [values[0]],
    theta=factors + [factors[0]],
    fill='toself',
    name='Risk Factors'
))
radar_fig.update_layout(
    polar=dict(radialaxis=dict(visible=True, range=[0,1])),
    showlegend=False
)
st.plotly_chart(radar_fig, use_container_width=True)

# 7. Bar chart of probability
st.subheader("Risk Probability Bar Chart")
prob_fig = go.Figure(go.Bar(
    x=["NAFPD Risk Probability"],
    y=[probability*100],
    text=[f"{probability*100:.1f}%"],
    textposition='outside'
))
prob_fig.update_layout(yaxis=dict(range=[0,100]))
st.plotly_chart(prob_fig, use_container_width=True)

# 8. Nomogram-like contribution chart
st.subheader("Variable Contribution (Nomogram Simulation)")
weights = [0.748, 0.903, 0.510, 0.721, 0.589, 0.731, 0.458]
contributions = [w * v for w, v in zip(weights, values)]
contrib_fig = go.Figure(go.Bar(
    y=factors,
    x=contributions,
    orientation='h',
    text=[f"{c:.2f}" for c in contributions],
    textposition='outside'
))
contrib_fig.update_layout(xaxis_title="Logit Contribution", title="Variable Contributions to Risk Prediction")
st.plotly_chart(contrib_fig, use_container_width=True)

# 9. PDF export using xhtml2pdf (pure Python)
def generate_pdf_report(age, fpg, ggt, waist, bmi, triglycerides, nlr, ast, alt, platelet, fli, mfib4, probability):
    html = f"""
    <html>
    <head>
        <style>
            body {{ font-family: Arial, sans-serif; padding: 20px; }}
            table {{ border-collapse: collapse; width: 100%; }}
            th, td {{ border: 1px solid #999; padding: 8px; text-align: center; }}
            th {{ background-color: #eee; }}
        </style>
    </head>
    <body>
        <h2>NAFPD Risk Assessment Report</h2>
        <p>Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
        <h3>Input Parameters</h3>
        <table>
            <tr><th>Parameter</th><th>Value</th></tr>
            <tr><td>Age</td><td>{age}</td></tr>
            <tr><td>FPG</td><td>{fpg}</td></tr>
            <tr><td>GGT</td><td>{ggt}</td></tr>
            <tr><td>Waist Circumference</td><td>{waist}</td></tr>
            <tr><td>BMI</td><td>{bmi}</td></tr>
            <tr><td>Triglycerides</td><td>{triglycerides}</td></tr>
            <tr><td>NLR</td><td>{nlr}</td></tr>
            <tr><td>AST</td><td>{ast}</td></tr>
            <tr><td>ALT</td><td>{alt}</td></tr>
            <tr><td>Platelet Count</td><td>{platelet}</td></tr>
        </table>
        <h3>Computed Indices</h3>
        <p>FLI: {fli:.2f}</p>
        <p>mFIB-4: {mfib4:.2f}</p>
        <p><strong>Risk Probability:</strong> {probability*100:.1f}%</p>
        <p style='color:red;'>Note: This report is for academic reference only and does not constitute clinical advice.</p>
    </body>
    </html>
    """
    pdf_bytes = BytesIO()
    pisa.CreatePDF(BytesIO(html.encode("utf-8")), dest=pdf_bytes)
    b64 = base64.b64encode(pdf_bytes.getvalue()).decode()
    return f"<a href='data:application/pdf;base64,{b64}' download='NAFPD_Report.pdf'>📄 Download PDF Report</a>"

# 10. Export PDF button
st.markdown("---")
st.subheader("Export Report")
pdf_link = generate_pdf_report(age, fpg, ggt, waist, bmi, triglycerides, nlr, ast, alt, platelet, fli, mfib4, probability)
st.markdown(pdf_link, unsafe_allow_html=True)
