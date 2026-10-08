import json
from pathlib import Path

import joblib
import pandas as pd
import requests
import streamlit as st

BASE = Path(__file__).resolve().parent
MODEL_URL = (
    "https://github.com/olohi184/WeatherImpactML/"
    "releases/download/v1.0.0/weatherimpactml_model.joblib"
)

st.set_page_config(
    page_title="WeatherImpactML | AI Weather Intelligence",
    page_icon="🌤️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
:root { color-scheme: dark; }
.stApp {
    background: radial-gradient(ellipse at 75% 0%, #15355c 0%, #0a1930 38%, #071222 85%);
    color: #e9f2ff;
}
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #102b4a, #08172a);
    border-right: 1px solid #294866;
}
[data-testid="stSidebar"] * { color: #e9f2ff; }
.block-container { padding-top: 1.8rem; max-width: 1320px; }
h1, h2, h3, p, label { color: #e9f2ff; }
.hero {
    padding: 32px 36px;
    border: 1px solid #2b5374;
    border-radius: 22px;
    background: linear-gradient(115deg, #153e63 0%, #0e2744 60%, #122f4e 100%);
    margin-bottom: 24px;
}
.eyebrow { color: #69d5ef; letter-spacing: .17em; font-size: .78rem; font-weight: 750; }
.hero h1 { font-size: clamp(2rem, 4vw, 3.2rem); margin: 9px 0 7px; color: #fff; }
.hero p { color: #b9d1e7; font-size: 1.05rem; margin: 0; }
.section-label { font-size: 1.16rem; font-weight: 750; margin: 18px 0 12px; color: #eaf6ff; }
.weather-card {
    padding: 22px 24px; border-radius: 18px;
    background: linear-gradient(150deg, #153653, #10243d);
    border: 1px solid #2c506c; min-height: 142px;
}
.card-icon { font-size: 1.45rem; }
.card-label { color: #a9c3dc; font-size: .9rem; margin-top: 9px; }
.card-value { color: #fff; font-size: clamp(1.65rem, 2.8vw, 2.4rem); font-weight: 750; }
.forecast-card {
    padding: 25px 30px; border-radius: 20px;
    background: linear-gradient(110deg, #0c5269, #12486a 55%, #163653);
    border: 1px solid #328ea8; margin: 12px 0 20px;
}
.forecast-value { color: #fff; font-size: clamp(2.8rem, 5vw, 4.5rem); font-weight: 800; }
.forecast-label { color: #b8f1fa; font-size: .92rem; letter-spacing: .08em; }
.forecast-note { color: #d4e9f6; font-size: .9rem; }
.metric-card {
    background: #10283f; border: 1px solid #2a4d68; border-radius: 15px;
    padding: 18px 21px; margin-bottom: 12px;
}
.metric-label { color: #a9c6dc; font-size: .87rem; }
.metric-value { color: #f2faff; font-size: 1.8rem; font-weight: 750; }
.small-note { color: #9cb8d0; font-size: .86rem; }
div.stButton > button[kind="primary"] {
    background: linear-gradient(90deg, #17a5cf, #3178e3);
    border: none; color: #fff; border-radius: 12px;
    font-weight: 750; padding: .7rem 1.5rem;
}
div[data-testid="stAlert"] { border-radius: 14px; }
[data-testid="stMetric"] { background: #10283f; border-radius: 12px; padding: 12px; }
hr { border-color: #31516b; }
</style>
""", unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_model():
    model_path = BASE / "weatherimpactml_model.joblib"
    if not model_path.exists():
        with requests.get(MODEL_URL, stream=True, timeout=120) as response:
            response.raise_for_status()
            with open(model_path, "wb") as output:
                for chunk in response.iter_content(chunk_size=1024 * 1024):
                    if chunk:
                        output.write(chunk)
    return joblib.load(model_path)


@st.cache_data
def load_data():
    with open(BASE / "weatherimpactml_metadata.json", encoding="utf-8") as f:
        metadata = json.load(f)
    data = pd.read_csv(BASE / "weatherimpactml_demo.csv")
    return metadata, data


def metric_card(label, value):
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div></div>',
        unsafe_allow_html=True,
    )


try:
    metadata, data = load_data()
    features = metadata["feature_names"]
    metrics = metadata["metrics"]
except Exception as exc:
    st.error(f"Unable to load project data: {exc}")
    st.stop()

with st.sidebar:
    st.markdown("## 🌤️ WeatherImpactML")
    st.caption("AI WEATHER INTELLIGENCE")
    st.divider()
    st.markdown("### Forecast controls")
    selected = st.selectbox(
        "Historical observation",
        range(len(data)),
        format_func=lambda i: f"Observation {i + 1}",
    )
    st.info("Select a historical weather observation, then generate its one-hour-ahead forecast.")
    st.divider()
    st.markdown("**Model** · Random Forest")
    st.markdown("**Forecast horizon** · 1 hour")
    st.markdown("**Data source** · Historical Jena weather observations")
    st.caption("Research demonstration · Not a live forecast")

observation = data.iloc[selected]

st.markdown("""
<div class="hero">
  <div class="eyebrow">MACHINE LEARNING • WEATHER ANALYTICS</div>
  <h1>🌤️ WeatherImpactML</h1>
  <p>Intelligent one-hour temperature prediction powered by a trained Random Forest model.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="section-label">Current observation</div>', unsafe_allow_html=True)
c1, c2, c3 = st.columns(3, gap="medium")
cards = [
    ("🌡️", "Temperature", f"{observation['T (degC)']:.2f} °C"),
    ("💧", "Relative humidity", f"{observation['rh (%)']:.1f}%"),
    ("🌬️", "Air pressure", f"{observation['p (mbar)']:.1f} mbar"),
]
for column, (symbol, label, value) in zip((c1, c2, c3), cards):
    with column:
        st.markdown(
            f'<div class="weather-card"><div class="card-icon">{symbol}</div>'
            f'<div class="card-label">{label}</div>'
            f'<div class="card-value">{value}</div></div>',
            unsafe_allow_html=True,
        )

st.write("")
if st.button("✨ Generate one-hour forecast", type="primary", use_container_width=True):
    try:
        with st.spinner("Running temperature prediction..."):
            model = load_model()
            input_data = pd.DataFrame(
                [[observation[name] for name in features]],
                columns=features,
            )
            predicted = float(model.predict(input_data)[0])
            actual = float(observation["actual_next_hour_c"])
            st.session_state["forecast"] = {
                "observation": selected,
                "predicted": predicted,
                "actual": actual,
            }
    except Exception as exc:
        st.error(f"Forecast unavailable: {exc}")

forecast = st.session_state.get("forecast")
if forecast and forecast["observation"] == selected:
    predicted = forecast["predicted"]
    actual = forecast["actual"]
    error = abs(actual - predicted)
    st.markdown(
        f'<div class="forecast-card">'
        f'<div class="forecast-label">PREDICTED TEMPERATURE · NEXT HOUR</div>'
        f'<div class="forecast-value">{predicted:.2f} °C</div>'
        f'<div class="forecast-note">Model estimate for the selected historical observation</div>'
        f'</div>',
        unsafe_allow_html=True,
    )
    left, right = st.columns(2, gap="medium")
    with left:
        metric_card("Actual next-hour temperature", f"{actual:.2f} °C")
    with right:
        metric_card("Absolute prediction error", f"{error:.2f} °C")

    st.markdown('<div class="section-label">Prediction comparison</div>', unsafe_allow_html=True)
    comparison = pd.DataFrame(
        {"Temperature (°C)": [predicted, actual]},
        index=["Predicted", "Actual"],
    )
    st.bar_chart(comparison, horizontal=True, color="#49cbe7")
else:
    st.markdown(
        '<p class="small-note">Generate a forecast to view the predicted temperature, '
        'observed outcome, and prediction comparison.</p>',
        unsafe_allow_html=True,
    )

st.divider()
st.markdown('<div class="section-label">Model performance · held-out historical test data</div>',
            unsafe_allow_html=True)
m1, m2, m3 = st.columns(3, gap="medium")
with m1:
    metric_card("Mean absolute error (MAE)", f"{metrics['MAE']:.3f} °C")
with m2:
    metric_card("Root mean squared error (RMSE)", f"{metrics['RMSE']:.3f} °C")
with m3:
    metric_card("R² score", f"{metrics['R2']:.4f}")

st.markdown(
    '<p class="small-note">WeatherImpactML is an educational and research demonstration '
    'using historical Jena weather data. Predictions are not live weather forecasts. '
    'Model metrics describe performance on the held-out historical test set.</p>',
    unsafe_allow_html=True,
)

