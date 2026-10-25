
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path

st.set_page_config(
    page_title="Air Quality Index Prediction",
    page_icon="🌍",
    layout="wide"
)

st.title("🌍 Air Quality Index (AQI) Prediction")
st.write("Environmental pollutant measurements tho AQI estimate cheyyadam.")

st.warning(
    "Educational project only. Ee prediction official air-quality reading "
    "kaadhu; health or emergency decisions kosam rely avvakandi."
)

model_path = Path("artifacts/aqi_model.joblib")
features_path = Path("artifacts/aqi_features.joblib")

if not model_path.exists() or not features_path.exists():
    st.error("Model files dorakaledhu. Mundhu python train.py run cheyyi.")
    st.stop()

model = joblib.load(model_path)
features = joblib.load(features_path)

st.subheader("Pollutant Measurements")

defaults = {
    "PM2.5": 30.0,
    "PM10": 50.0,
    "NO": 10.0,
    "NO2": 20.0,
    "NOx": 30.0,
    "NH3": 10.0,
    "CO": 0.5,
    "SO2": 10.0,
    "O3": 30.0,
    "Benzene": 1.0,
    "Toluene": 5.0,
    "Xylene": 1.0
}

inputs = {}

with st.form("aqi_form"):
    columns = st.columns(3)

    for i, feature in enumerate(features):
        with columns[i % 3]:
            inputs[feature] = st.number_input(
                feature,
                min_value=0.0,
                value=float(defaults.get(feature, 0.0)),
                step=0.1
            )

    submitted = st.form_submit_button("Predict AQI")

if submitted:
    input_df = pd.DataFrame([inputs], columns=features)
    prediction = float(model.predict(input_df)[0])
    prediction = max(0.0, prediction)

    st.subheader("Predicted AQI")
    st.metric("Estimated AQI", f"{prediction:.2f}")

    if prediction <= 50:
        st.info("Indicative category: Good")
    elif prediction <= 100:
        st.info("Indicative category: Satisfactory")
    elif prediction <= 200:
        st.warning("Indicative category: Moderate")
    elif prediction <= 300:
        st.warning("Indicative category: Poor")
    elif prediction <= 400:
        st.error("Indicative category: Very Poor")
    else:
        st.error("Indicative category: Severe")

    st.caption(
        "Category bands are indicative Indian AQI bands. "
        "This model estimate is not an official AQI measurement."
    )
    