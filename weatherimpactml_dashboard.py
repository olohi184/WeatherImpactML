import requests
import json
import joblib
import pandas as pd
import streamlit as st
from pathlib import Path

BASE = Path(__file__).resolve().parent

st.set_page_config(
    page_title="WeatherImpactML",
    page_icon="🌤️",
    layout="wide"
)

@st.cache_resource
def load_model():
    return joblib.load(BASE / "weatherimpactml_model.joblib")

model = @st.cache_resource
def load_model():
    model_path = BASE / "weatherimpactml_model.joblib"

    if not model_path.exists():
        url = (
            "https://github.com/olohi184/WeatherImpactML/"
            "releases/download/v1.0.0/weatherimpactml_model.joblib"
        )

        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(model_path, "wb") as f:
                for chunk in response.iter_content(1024 * 1024):
                    if chunk:
                        f.write(chunk)

    return joblib.load(model_path)

with open(BASE / "weatherimpactml_metadata.json") as f:
    metadata = json.load(f)

data = pd.read_csv(BASE / "weatherimpactml_demo.csv")
features = metadata["feature_names"]

st.title("🌤️ WeatherImpactML")
st.subheader("AI-Powered One-Hour Temperature Forecasting")

st.write(
    "An ML engineering demonstration using historical "
    "weather observations and a Random Forest model."
)

st.sidebar.header("Forecast Settings")

selected = st.sidebar.selectbox(
    "Select a weather observation",
    range(len(data)),
    format_func=lambda i: f"Observation {i + 1}"
)

observation = data.iloc[selected]

st.subheader("Weather Measurements")

col1, col2, col3 = st.columns(3)

col1.metric(
    "Current Temperature",
    f"{observation['T (degC)']:.2f} °C"
)

col2.metric(
    "Humidity",
    f"{observation['rh (%)']:.1f}%"
)

col3.metric(
    "Pressure",
    f"{observation['p (mbar)']:.1f} mbar"
)

if st.button("Generate Forecast", type="primary"):
    input_data = pd.DataFrame(
        [[observation[name] for name in features]],
        columns=features
    )

    prediction = float(model.predict(input_data)[0])
    actual = float(observation["actual_next_hour_c"])

    st.success(
        f"Predicted Temperature: {prediction:.2f} °C"
    )

    col_a, col_b = st.columns(2)

    col_a.metric(
        "Actual Next-Hour Temperature",
        f"{actual:.2f} °C"
    )

    col_b.metric(
        "Absolute Prediction Error",
        f"{abs(actual - prediction):.2f} °C"
    )

st.divider()

st.subheader("Model Performance")

metrics = metadata["metrics"]

c1, c2, c3 = st.columns(3)

c1.metric("MAE", f"{metrics['MAE']:.3f} °C")
c2.metric("RMSE", f"{metrics['RMSE']:.3f} °C")
c3.metric("R²", f"{metrics['R2']:.4f}")

st.caption(
    "Research and educational demonstration using "
    "historical Jena weather data. Not a live weather forecast."
)
