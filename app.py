import streamlit as st
import pandas as pd
import numpy as np
import joblib
import plotly.graph_objects as go
import os

st.set_page_config(page_title="Rainfall Forecast", page_icon="🌧️", layout="wide")

@st.cache_resource
def load_pipeline():
    if not os.path.exists("rainfall_pipeline.pkl"):
        return None
    return joblib.load("rainfall_pipeline.pkl")

pipeline = load_pipeline()
st.title("🌧️ Rainfall Forecasting & Heavy-Rain Risk Classifier")

if pipeline is None:
    st.error("Model file not found! Please run 'python train_model.py' first in your terminal.")
    st.stop()

scaler = pipeline['scaler']
regressor = pipeline['regressor']
classifier = pipeline['classifier']
feature_cols = pipeline['features']
class_labels = pipeline['classes']

col1, col2, col3 = st.columns(3)
with col1:
    temp = st.slider("Temperature (°C)", 10.0, 48.0, 28.0, 0.5)
    humidity = st.slider("Humidity (%)", 20.0, 100.0, 82.0, 1.0)
with col2:
    pressure = st.slider("Pressure (hPa)", 960.0, 1035.0, 1003.0, 0.5)
    wind_speed = st.slider("Wind Speed (km/h)", 0.0, 85.0, 24.0, 1.0)
with col3:
    cloud_cover = st.slider("Cloud Cover (%)", 0.0, 100.0, 80.0, 1.0)
    dew_point = st.slider("Dew Point (°C)", 5.0, 35.0, 25.0, 0.5)

input_df = pd.DataFrame([[temp, humidity, pressure, wind_speed, cloud_cover, dew_point]], columns=feature_cols)
scaled_input = scaler.transform(input_df)

pred_rainfall = float(regressor.predict(scaled_input)[0])
pred_risk = classifier.predict(scaled_input)[0]
prob_dist = classifier.predict_proba(scaled_input)[0]

st.markdown("---")
res1, res2 = st.columns(2)
with res1:
    st.metric("Estimated Rainfall", f"{pred_rainfall:.2f} mm")
    if "Extreme" in pred_risk:
        st.error(f"Hazard Alert: **{pred_risk}**")
    elif "Heavy" in pred_risk:
        st.warning(f"Hazard Alert: **{pred_risk}**")
    else:
        st.info(f"Hazard Alert: **{pred_risk}**")

with res2:
    fig = go.Figure(go.Bar(x=[p * 100 for p in prob_dist], y=class_labels, orientation='h'))
    fig.update_layout(xaxis_title="Confidence (%)", height=240, margin=dict(l=10, r=10, t=10, b=10))
    st.plotly_chart(fig, use_container_width=True)