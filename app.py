
import os
import streamlit as st
import pandas as pd
from tabpfn_client import TabPFNClassifier

# ==================================================
# PAGE CONFIGURATION
# ==================================================
st.set_page_config(
    page_title="CropWise AI | Smart Agriculture",
    page_icon="🌱",
    layout="wide",
    initial_sidebar_state="expanded"
)

FEATURES = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

# ==================================================
# DASHBOARD DESIGN
# ==================================================
st.markdown("""
<style>
.stApp {
    background: #f5f8f3;
}
[data-testid="stSidebar"] {
    background: #e9f2e6;
}
.main-title {
    color: #205c36;
    font-size: 2.3rem;
    font-weight: 800;
    margin-bottom: 0;
}
.subtitle {
    color: #607564;
    margin-bottom: 20px;
}
div[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce8d7;
    padding: 14px;
    border-radius: 12px;
}
div.stButton > button,
div[data-testid="stFormSubmitButton"] > button {
    background: #287a43;
    color: white;
    border-radius: 9px;
    border: none;
    min-height: 44px;
    font-weight: 600;
}
div.stButton > button:hover,
div[data-testid="stFormSubmitButton"] > button:hover {
    background: #205c36;
    color: white;
}
</style>
""", unsafe_allow_html=True)

# ==================================================
# LOAD DATASET
# ==================================================
@st.cache_data
def load_dataset():
    data = pd.read_csv("crop_data.csv")
    data.columns = data.columns.str.strip()

    required = FEATURES + ["label"]
    missing = [c for c in required if c not in data.columns]

    if missing:
        raise ValueError(
            "Missing required dataset columns: "
            + ", ".join(missing)
        )

    for column in FEATURES:
        data[column] = pd.to_numeric(
            data[column], errors="coerce"
        )

    data = data.dropna(
        subset=FEATURES + ["label"]
    ).copy()

    data["label"] = data["label"].astype(str).str.strip()
    data = data[data["label"] != ""]

    if data.empty:
        raise ValueError("No valid training records were found.")

    return data


try:
    df = load_dataset()
except Exception as error:
    st.error("Could not load crop_data.csv.")
    st.code(str(error))
    st.stop()

# ==================================================
# HEADER
# ==================================================
st.markdown(
    '<div class="main-title">🌱 CropWise AI</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">'
    'Smart crop recommendations and agricultural insights'
    '</div>',
    unsafe_allow_html=True
)

# ==================================================
# SIDEBAR: FARM INPUTS
# ==================================================
st.sidebar.title("🌾 Farm Conditions")
st.sidebar.write("Enter your soil and weather measurements.")

with st.sidebar.form("farm_conditions"):
    N = st.number_input(
        "Nitrogen (N)", 0.0, 200.0, 90.0, step=1.0
    )
    P = st.number_input(
        "Phosphorus (P)", 0.0, 200.0, 42.0, step=1.0
    )
    K = st.number_input(
        "Potassium (K)", 0.0, 200.0, 43.0, step=1.0
    )
    temperature = st.number_input(
        "Temperature (°C)", -10.0, 60.0, 20.88, step=0.5
    )
    humidity = st.number_input(
        "Humidity (%)", 0.0, 100.0, 82.0, step=1.0
    )
    ph = st.number_input(
        "Soil pH", 0.0, 14.0, 6.5, step=0.1
    )
    rainfall = st.number_input(
        "Rainfall (mm)", 0.0, 1000.0, 202.94, step=1.0
    )

    submitted = st.form_submit_button(
        "🌱 Recommend Crop",
        use_container_width=True
    )

farm_input = {
    "N": N,
    "P": P,
    "K": K,
    "temperature": temperature,
    "humidity": humidity,
    "ph": ph,
    "rainfall": rainfall
}

# ==================================================
# RUN TABPFN
# ==================================================
if submitted:
    st.session_state.pop("crop_result", None)
    st.session_state.pop("crop_error", None)

    try:
        if "TABPFN_TOKEN" not in st.secrets:
            raise ValueError(
                "TABPFN_TOKEN is missing from Streamlit Secrets."
            )

        os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]

        with st.spinner("TabPFN is analysing your farm conditions..."):
            X = df[FEATURES]
            y = df["label"]

            if y.nunique() < 2:
                raise ValueError(
                    "At least two crop categories are required."
                )

            model = TabPFNClassifier()
            model.fit(X, y)

            new_input = pd.DataFrame(
                [farm_input],
                columns=FEATURES
            )

            prediction = str(model.predict(new_input)[0])
            probabilities = model.predict_proba(new_input)[0]
            classes = model.classes_

            top_indices = sorted(
                range(len(probabilities)),
                key=lambda i: probabilities[i],
                reverse=True
            )[:3]

            recommendations = [
                {
                    "crop": str(classes[i]),
                    "confidence": float(probabilities[i]) * 100
                }
                for i in top_indices
            ]

            st.session_state["crop_result"] = {
                "prediction": prediction,
                "recommendations": recommendations,
                "inputs": farm_input.copy()
            }

    except Exception as error:
        st.session_state["crop_error"] = str(error)

if st.session_state.get("crop_error"):
    st.error("The recommendation could not be generated.")
    st.code(st.session_state["crop_error"])

result = st.session_state.get("crop_result")

# ==================================================
# DASHBOARD SUMMARY
# ==================================================
price_records = 0

if "price_per_quintal" in df.columns:
    prices = pd.to_numeric(
        df["price_per_quintal"], errors="coerce"
    )
    price_records = int(prices.notna().sum())

c1, c2, c3 = st.columns(3)
c1.metric("Dataset Records", len(df))
c2.metric("Crop Categories", df["label"].nunique())
c3.metric("Stored Price Records", price_records)

st.divider()

# ==================================================
# DASHBOARD TABS
# ==================================================
tab_crop, tab_soil, tab_weather, tab_fertilizer, tab_market, tab_analytics = st.tabs([
    "🌱 Crop AI",
    "🧪 Soil Health",
    "🌦️ Weather",
    "🧴 Fertilizer",
    "💹 Market",
    "📊 Analytics"
])

# ==================================================
# 1. CROP AI
# ==================================================
with tab_crop:
    st.subheader("AI Crop Recommendation")

    if result:
        prediction = result["prediction"]
        recommendations = result["recommendations"]

        st.success(f"Recommended Crop: {prediction.upper()}")

        left, right = st.columns(2)

        with left:
            st.markdown("### Top 3 Recommendations")

            for rank, item in enumerate(recommendations, start=1):
                st.write(
                    f"**{rank}. {item['crop'].upper()}** — "
                    f"{item['confidence']:.2f}% model confidence"
                )
                st.progress(
                    max(0.0, min(item["confidence"] / 100, 1.0))
                )

        with right:
            st.markdown("### Crop Information")

            crop_rows = df[
                df["label"].str.lower() == prediction.lower()
            ]

            if not crop_rows.empty:
                info = crop_rows.iloc[0]

                for column, title in [
                    ("season", "Season"),
                    ("soil_type", "Dataset Soil Type"),
                    ("water_need", "Dataset Water Need"),
                    ("state", "Dataset State")
                ]:
                    if column in df.columns and pd.notna(info.get(column)):
                        st.write(f"**{title}:** {info[column]}")

                if (
                    "price_per_quintal" in df.columns
                    and pd.notna(info.get("price_per_quintal"))
                ):
                    try:
                        price = float(info["price_per_quintal"])
                        st.write(
                            f"**Stored price:** ₹{price:,.0f} per quintal"
                        )
                    except (ValueError, TypeError):
                        pass

        st.info(
            "Model confidence is not the probability of crop success, "
            "yield or profit in real farming conditions."
        )

    else:
        st.info(
            "Enter the farm conditions in the sidebar and press "
            "'Recommend Crop' to start."
        )

    with st.expander("Understand the seven input values"):
        st.write("""
        - **N:** Nitrogen measurement
        - **P:** Phosphorus measurement
        - **K:** Potassium measurement
        - **Temperature:** Air temperature in °C
        - **Humidity:** Relative humidity in %
        - **pH:** Soil acidity or alkalinity measurement
        - **Rainfall:** Rainfall in millimetres

        Use correctly measured values and the units expected by
        your dataset for meaningful model predictions.
        """)

# ==================================================
# 2. SOIL HEALTH
# ==================================================
with tab_soil:
    st.subheader("Soil Condition Overview")
    st.write(
        "Review the measurements entered for the crop model. "
        "These are not laboratory-confirmed soil diagnoses."
    )

    a, b, c = st.columns(3)
    a.metric("Nitrogen (N)", f"{N:g}")
    b.metric("Phosphorus (P)", f"{P:g}")
    c.metric("Potassium (K)", f"{K:g}")

    d, e = st.columns(2)
    d.metric("Soil pH", f"{ph:.1f}")
    e.metric("Rainfall Input", f"{rainfall:g} mm")

    if ph < 5.5:
        st.warning(
            "The entered pH is acidic. Confirm with a soil test "
            "before choosing any soil treatment."
        )
    elif ph > 7.5:
        st.warning(
            "The entered pH is alkaline. Confirm with a soil test "
            "before choosing any soil treatment."
        )
    else:
        st.info(
            "This pH is within a range suitable for many crops, "
            "but the ideal range depends on the specific crop."
        )

    soil_table = pd.DataFrame({
        "Measurement": [
            "Nitrogen", "Phosphorus", "Potassium", "Soil pH",
            "Temperature", "Humidity", "Rainfall"
        ],
        "Entered Value": [
            N, P, K, ph, temperature, humidity, rainfall
        ]
    })

    st.dataframe(
        soil_table, use_container_width=True, hide_index=True
    )

    st.caption(
        "NPK interpretation requires known units and a reliable soil "
        "testing method. Consult a soil laboratory for advice."
    )

# ==================================================
# 3. WEATHER
# ==================================================
with tab_weather:
    st.subheader("Weather & Growing Conditions")
    st.caption(
        "These are the values entered for the AI prediction, "
        "not a live weather forecast."
    )

    a, b, c = st.columns(3)
    a.metric("Temperature", f"{temperature:.1f} °C")
    b.metric("Humidity", f"{humidity:.1f}%")
    c.metric("Rainfall", f"{rainfall:.1f} mm")

    weather_table = pd.DataFrame({
        "Condition": ["Temperature", "Humidity", "Rainfall"],
        "Entered Value": [temperature, humidity, rainfall]
    }).set_index("Condition")

    st.bar_chart(weather_table)

    st.info(
        "Live weather requires a weather data service and a location. "
        "This prototype does not fetch live weather."
    )

# ==================================================
# 4. FERTILIZER
# ==================================================
with tab_fertilizer:
    st.subheader("Fertilizer & Nutrient Guidance")

    a, b, c = st.columns(3)
    a.metric("Nitrogen Input", f"{N:g}")
    b.metric("Phosphorus Input", f"{P:g}")
    c.metric("Potassium Input", f"{K:g}")

    if result:
        st.write(
            f"**Recommended crop to investigate:** "
            f"{result['prediction'].upper()}"
        )

    st.markdown("""
    ### Suggested next steps

    1. Obtain a soil test and confirm the nutrient units.
    2. Check the nutrient needs of the selected crop.
    3. Consider crop stage, irrigation and previous fertilizer use.
    4. Follow local agricultural department recommendations and
       the fertilizer product label.
    """)

    st.warning(
        "This prototype does not prescribe fertilizer quantities. "
        "NPK values alone are insufficient for a reliable dosage."
    )

# ==================================================
# 5. MARKET
# ==================================================
with tab_market:
    st.subheader("Crop Market Information")

    if "price_per_quintal" not in df.columns:
        st.info(
            "The dataset does not contain price_per_quintal data."
        )
    else:
        market_df = df.copy()
        market_df["price_per_quintal"] = pd.to_numeric(
            market_df["price_per_quintal"], errors="coerce"
        )
        market_df = market_df.dropna(
            subset=["price_per_quintal"]
        )

        if market_df.empty:
            st.info("No valid stored price records were found.")
        else:
            summary = (
                market_df.groupby("label")["price_per_quintal"]
                .agg(["mean", "min", "max", "count"])
                .reset_index()
                .rename(columns={
                    "label": "Crop",
                    "mean": "Average Price (₹/quintal)",
                    "min": "Minimum Price (₹/quintal)",
                    "max": "Maximum Price (₹/quintal)",
                    "count": "Records"
                })
                .round(2)
            )

            st.dataframe(
                summary, use_container_width=True, hide_index=True
            )

            if result:
                selected = market_df[
                    market_df["label"].str.lower()
                    == result["prediction"].lower()
                ]

                if not selected.empty:
                    avg_price = selected["price_per_quintal"].mean()
                    st.metric(
                        "Average Stored Price for Recommended Crop",
                        f"₹{avg_price:,.0f}/quintal"
                    )

            st.caption(
                "Prices come from crop_data.csv. They are stored dataset "
                "values, not verified live mandi prices. Check the local "
                "market and date before making financial decisions."
            )

    for column, title in [
        ("market", "Market"),
        ("state", "State"),
        ("last_updated", "Dataset Update Date")
    ]:
        if column in df.columns:
            st.write(f"Dataset field available: **{title}**")

# ==================================================
# 6. ANALYTICS
# ==================================================
with tab_analytics:
    st.subheader("Dataset Analytics")

    st.markdown("### Crop Distribution")
    distribution = (
        df["label"].value_counts()
        .rename_axis("Crop")
        .reset_index(name="Records")
        .set_index("Crop")
    )
    st.bar_chart(distribution)

    a, b, c = st.columns(3)
    a.metric("Valid Dataset Rows", len(df))
    b.metric("Crop Categories", df["label"].nunique())
    c.metric(
        "Missing Model Inputs",
        int(df[FEATURES].isna().sum().sum())
    )

    if "price_per_quintal" in df.columns:
        price_data = df.copy()
        price_data["price_per_quintal"] = pd.to_numeric(
            price_data["price_per_quintal"], errors="coerce"
        )

        average_prices = (
            price_data.dropna(subset=["price_per_quintal"])
            .groupby("label")["price_per_quintal"]
            .mean()
            .sort_values(ascending=False)
        )

        if not average_prices.empty:
            st.markdown("### Average Stored Price by Crop")
            st.bar_chart(average_prices)

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True)

    st.caption(
        "Analytics describe the supplied dataset. They do not establish "
        "accuracy or represent every farm or market."
    )

# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "CropWise AI | Academic Capstone Prototype | "
    "Python + Streamlit + TabPFN"
)
