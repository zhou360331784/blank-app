import streamlit as st
import numpy as np
import plotly.graph_objects as go
# from xhtml2pdf import pisa
from io import BytesIO
import base64
from datetime import datetime

# Configure page
st.set_page_config(page_title="FPD Risk Assessment", layout="centered")

st.title("Fatty Pancreas Disease (FPD) Risk Assessment Tool")

# Section 1: Select Gender
st.header("Section 1: Select Gender")
gender = st.selectbox("Gender", ["", "Male", "Female"], format_func=lambda x: "-- Select --" if x=="" else x)
if gender == "":
    st.warning("Please select gender to proceed.")
    st.stop()

# Section 2: Enter Clinical Parameters
st.header("Section 2: Enter Clinical Parameters")
st.markdown(f"**Selected Gender:** {gender}")
with st.form("clinical_form"):
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("Age (years)", 1, 120, 50)
        fpg = st.number_input("Fasting Plasma Glucose (mmol/L)", 2.0, 30.0, 5.5)
        ggt = st.number_input("GGT (U/L)", 5.0, 500.0, 30.0)
        waist = st.number_input("Waist Circumference (cm)", 30.0, 150.0, 85.0)
        nlr = st.number_input("Neutrophil/Lymphocyte Ratio (NLR)", 0.1, 15.0, 1.5)
    with col2:
        triglycerides = st.number_input("Triglycerides (mg/dL)", 10.0, 1000.0, 150.0)
        bmi = st.number_input("Body Mass Index (kg/m²)", 10.0, 50.0, 23.0)
        ast = st.number_input("AST (U/L)", 5.0, 500.0, 20.0)
        alt = st.number_input("ALT (U/L)", 5.0, 500.0, 25.0)
        platelet = st.number_input("Platelet Count (10⁹/L)", 10.0, 1000.0, 200.0)
    submit = st.form_submit_button("Calculate Risk")

if not submit:
    st.stop()

# Compute indices
logit_fli = 0.953 * np.log(triglycerides) + 0.139 * bmi + 0.718 * np.log(ggt) + 0.053 * waist - 15.745
fli = np.exp(logit_fli) / (1 + np.exp(logit_fli)) * 100
mfib4 = 10 * age * ast / (platelet * alt)

# Gender-specific thresholds
waist_thresh = 93.4 if gender == "Male" else 88.493
ggt_thresh = 50.0 if gender == "Male" else 32.0

# Binary encoding
x = [age > 65, fpg > 6.1, ggt > ggt_thresh, waist > waist_thresh, fli > 24.7, mfib4 > 3.05, nlr > 1.97]
coeffs = [0.748, 0.903, 0.510, 0.721, 0.589, 0.731, 0.458]

# Calculate risk
intercept = -3.089
logit_p = intercept + sum(c * v for c, v in zip(coeffs, x))
prob = 1 / (1 + np.exp(-logit_p))

# Display results
st.subheader("Results")
st.write(f"FLI: {fli:.2f}")
st.write(f"mFIB-4: {mfib4:.2f}")
st.write(f"Waist Threshold ({waist_thresh} cm): {'High' if x[3] else 'Normal'}")
st.write(f"GGT Threshold ({ggt_thresh} U/L): {'High' if x[2] else 'Normal'}")
st.write(f"Estimated Risk Probability: {prob*100:.1f}%")
if prob < 0.2:
    st.success("Low risk")
elif prob < 0.5:
    st.warning("Moderate risk")
else:
    st.error("High risk")

# Visualizations
factors = ['Age>65','FPG>6.1','GGT','Waist','FLI>24.7','mFIB4>3.05','NLR>1.97']
values = list(map(int, x))

st.subheader("Risk Factor Radar Chart")
radar = go.Figure(go.Scatterpolar(r=values+[values[0]], theta=factors+[factors[0]], fill='toself'))
radar.update_layout(polar=dict(radialaxis=dict(range=[0,1])), showlegend=False)
st.plotly_chart(radar, use_container_width=True)

st.subheader("Risk Probability Bar Chart")
bar = go.Figure(go.Bar(x=["Risk Probability"], y=[prob*100], text=[f"{prob*100:.1f}%"], textposition='outside'))
bar.update_layout(yaxis=dict(range=[0,100]))
st.plotly_chart(bar, use_container_width=True)

st.subheader("Variable Contribution Chart")
contrib = [c * v for c, v in zip(coeffs, values)]
chart = go.Figure(go.Bar(y=factors, x=contrib, orientation='h', text=[f"{c:.2f}" for c in contrib], textposition='outside'))
chart.update_layout(xaxis_title="Contribution")
st.plotly_chart(chart, use_container_width=True)
