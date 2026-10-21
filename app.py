
import os
import streamlit as st
import pandas as pd
from tabpfn_client import TabPFNClassifier

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="CropWise AI",
    page_icon="🌱",
    layout="wide"
)

st.markdown("""
<style>
.main {
    background-color: #f7faf7;
}
h1, h2, h3 {
    color: #205c39;
}
[data-testid="stMetric"] {
    background: white;
    border: 1px solid #dce9dd;
    padding: 14px;
    border-radius: 12px;
}
.stButton button {
    background-color: #247747;
    color: white;
    border-radius: 9px;
    font-weight: 600;
}
</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# LOAD SECRET AND DATASET
# --------------------------------------------------
try:
    os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
except Exception:
    st.error(
        "TabPFN secret is missing. Add TABPFN_TOKEN "
        "in Streamlit app settings."
    )
    st.stop()

try:
    df = pd.read_csv("crop_data.csv")
except Exception as e:
    st.error("Could not load crop_data.csv.")
    st.code(str(e))
    st.stop()

required_columns = [
    "N", "P", "K", "temperature", "humidity",
    "ph", "rainfall", "label", "price_per_quintal",
    "season", "soil_type", "water_need", "state"
]

missing = [c for c in required_columns if c not in df.columns]

if missing:
    st.error(f"Dataset is missing columns: {missing}")
    st.stop()

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]

df["label"] = df["label"].astype(str).str.strip()
df["price_per_quintal"] = pd.to_numeric(
    df["price_per_quintal"], errors="coerce"
)

df = df.dropna(subset=features + ["label"])

# --------------------------------------------------
# HEADER
# --------------------------------------------------
st.title("🌱 CropWise AI")
st.write(
    "Smart crop recommendations and agricultural insights"
)
st.caption(
    "Academic capstone prototype | Python + Streamlit + TabPFN"
)

m1, m2, m3 = st.columns(3)
m1.metric("Dataset Records", len(df))
m2.metric("Crop Categories", df["label"].nunique())
m3.metric(
    "Stored Price Records",
    int(df["price_per_quintal"].notna().sum())
)

st.divider()

# --------------------------------------------------
# SIX DASHBOARD SECTIONS
# --------------------------------------------------
crop_tab, soil_tab, weather_tab, fertilizer_tab, market_tab, analytics_tab = (
    st.tabs([
        "🌾 Crop AI",
        "🧪 Soil Health",
        "🌦️ Weather",
        "🌱 Fertilizer",
        "💰 Market",
        "📊 Analytics"
    ])
)

# --------------------------------------------------
# CROP AI
# --------------------------------------------------
with crop_tab:
    st.subheader("AI Crop Recommendation")
    st.write("Enter the soil and weather conditions for your field.")

    c1, c2, c3 = st.columns(3)

    with c1:
        N = st.number_input(
            "Nitrogen (N)", 0.0, 200.0, 90.0, key="input_N"
        )
        P = st.number_input(
            "Phosphorus (P)", 0.0, 200.0, 42.0, key="input_P"
        )
        K = st.number_input(
            "Potassium (K)", 0.0, 200.0, 43.0, key="input_K"
        )

    with c2:
        temperature = st.number_input(
            "Temperature (°C)", -10.0, 60.0, 20.88,
            key="input_temperature"
        )
        humidity = st.number_input(
            "Humidity (%)", 0.0, 100.0, 82.0,
            key="input_humidity"
        )

    with c3:
        ph = st.number_input(
            "Soil pH", 0.0, 14.0, 6.5, key="input_ph"
        )
        rainfall = st.number_input(
            "Rainfall (mm)", 0.0, 1000.0, 202.94,
            key="input_rainfall"
        )

    input_data = {
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }

    if st.button("🌱 Recommend Crop", use_container_width=True):
        try:
            with st.spinner("TabPFN is analyzing the inputs..."):
                X = df[features]
                y = df["label"]

                model = TabPFNClassifier()
                model.fit(X, y)

                new_input = pd.DataFrame([input_data], columns=features)
                prediction = str(model.predict(new_input)[0])

                probabilities = model.predict_proba(new_input)[0]
                classes = model.classes_
                top_indices = probabilities.argsort()[-3:][::-1]

                st.session_state["crop_result"] = prediction
                st.session_state["crop_top3"] = [
                    (str(classes[i]), float(probabilities[i]) * 100)
                    for i in top_indices
                ]

        except Exception as e:
            st.error("The TabPFN prediction could not run.")
            st.code(str(e))

    prediction = st.session_state.get("crop_result")

    if prediction:
        st.success(f"Recommended Crop: {prediction.upper()}")

        st.subheader("Top 3 AI Recommendations")
        for rank, (crop, confidence) in enumerate(
            st.session_state.get("crop_top3", []), start=1
        ):
            st.write(
                f"**{rank}. {crop.upper()} — "
                f"{confidence:.2f}% model confidence**"
            )
            st.progress(max(0.0, min(confidence / 100, 1.0)))

        crop_rows = df[
            df["label"].str.lower() == prediction.strip().lower()
        ]

        if not crop_rows.empty:
            st.subheader("Crop Information")

            info1, info2 = st.columns(2)
            row = crop_rows.iloc[0]

            info1.write(f"**Season:** {row['season']}")
            info1.write(f"**Dataset soil type:** {row['soil_type']}")
            info2.write(f"**Water need:** {row['water_need']}")
            info2.write(f"**Dataset state:** {row['state']}")

            # Use the numeric mean of all stored prices for this crop.
            crop_prices = pd.to_numeric(
                crop_rows["price_per_quintal"], errors="coerce"
            ).dropna()

            if not crop_prices.empty:
                average_price = float(crop_prices.mean())
                st.metric(
                    "Average Stored Price for Recommended Crop",
                    f"₹{average_price:,.0f}/quintal"
                )
            else:
                st.info("No stored price is available for this crop.")

        st.caption(
            "Model confidence is not a guarantee of crop success, "
            "yield, or profit. Results depend on dataset quality."
        )
    else:
        st.info("Enter the field conditions and tap Recommend Crop.")

# --------------------------------------------------
# SOIL HEALTH
# --------------------------------------------------
with soil_tab:
    st.subheader("Soil Health Overview")
    st.write("Review the soil measurements entered in Crop AI.")

    s1, s2, s3 = st.columns(3)
    s1.metric("Nitrogen (N)", f"{N:.1f}")
    s2.metric("Phosphorus (P)", f"{P:.1f}")
    s3.metric("Potassium (K)", f"{K:.1f}")

    s4, s5 = st.columns(2)
    s4.metric("Soil pH", f"{ph:.1f}")
    s5.metric("Rainfall", f"{rainfall:.1f} mm")

    soil_chart = pd.DataFrame({
        "Nutrient": ["Nitrogen", "Phosphorus", "Potassium"],
        "Entered value": [N, P, K]
    }).set_index("Nutrient")

    st.subheader("NPK Measurement Chart")
    st.bar_chart(soil_chart)

    st.info(
        "These are entered measurements, not a laboratory soil diagnosis. "
        "Confirm units and soil condition with a soil test."
    )

# --------------------------------------------------
# WEATHER
# --------------------------------------------------
with weather_tab:
    st.subheader("Weather Conditions")

    w1, w2, w3 = st.columns(3)
    w1.metric("Temperature", f"{temperature:.1f} °C")
    w2.metric("Humidity", f"{humidity:.1f}%")
    w3.metric("Rainfall", f"{rainfall:.1f} mm")

    weather_chart = pd.DataFrame({
        "Measurement": ["Temperature", "Humidity", "Rainfall"],
        "Entered value": [temperature, humidity, rainfall]
    }).set_index("Measurement")

    st.bar_chart(weather_chart)

    st.info(
        "This prototype displays user-entered values. "
        "It does not fetch live weather or provide a forecast."
    )

# --------------------------------------------------
# FERTILIZER
# --------------------------------------------------
with fertilizer_tab:
    st.subheader("Fertilizer and Nutrient Guidance")

    prediction = st.session_state.get("crop_result")

    if prediction:
        st.write(f"**Recommended crop to investigate:** {prediction.upper()}")
    else:
        st.write("Run a crop recommendation to select a crop for investigation.")

    f1, f2, f3 = st.columns(3)
    f1.metric("Nitrogen (N)", f"{N:.1f}")
    f2.metric("Phosphorus (P)", f"{P:.1f}")
    f3.metric("Potassium (K)", f"{K:.1f}")

    st.markdown("### Suggested next steps")
    st.markdown("""
    1. Obtain a soil test and confirm nutrient units.
    2. Check the nutrient needs of the selected crop.
    3. Consider crop stage, irrigation and previous fertilizer use.
    4. Follow local agricultural department recommendations and
       the fertilizer product label.
    """)

    st.warning(
        "This prototype does not prescribe fertilizer quantities. "
        "NPK values alone are insufficient for a reliable dosage."
    )

# --------------------------------------------------
# MARKET
# --------------------------------------------------
with market_tab:
    st.subheader("Stored Crop Price Comparison")

    market = (
        df.dropna(subset=["price_per_quintal"])
        .groupby("label")["price_per_quintal"]
        .agg(["mean", "min", "max"])
        .reset_index()
    )

    market.columns = [
        "Crop", "Average Price (₹/quintal)",
        "Minimum Price (₹/quintal)", "Maximum Price (₹/quintal)"
    ]

    market = market.sort_values(
        "Average Price (₹/quintal)", ascending=False
    )

    display_market = market.copy()
    for column in [
        "Average Price (₹/quintal)",
        "Minimum Price (₹/quintal)",
        "Maximum Price (₹/quintal)"
    ]:
        display_market[column] = display_market[column].map(
            lambda x: f"₹{x:,.0f}"
        )

    st.dataframe(display_market, use_container_width=True, hide_index=True)

    if prediction:
        selected = df[
            df["label"].str.lower() == prediction.strip().lower()
        ]
        prices = pd.to_numeric(
            selected["price_per_quintal"], errors="coerce"
        ).dropna()

        if not prices.empty:
            st.metric(
                "Average Stored Price for Recommended Crop",
                f"₹{prices.mean():,.0f}/quintal"
            )

    st.bar_chart(
        market.set_index("Crop")[["Average Price (₹/quintal)"]]
    )

    st.caption(
        "Prices come from crop_data.csv. They are stored dataset values, "
        "not verified live mandi prices. Check the market and date before "
        "making financial decisions."
    )

# --------------------------------------------------
# ANALYTICS
# --------------------------------------------------
with analytics_tab:
    st.subheader("Dataset Analytics")

    counts = (
        df["label"].value_counts()
        .rename_axis("Crop")
        .reset_index(name="Records")
    )

    a1, a2 = st.columns(2)
    a1.metric("Total Valid Records", len(df))
    a2.metric("Unique Crop Categories", df["label"].nunique())

    st.markdown("### Records by Crop")
    st.bar_chart(counts.set_index("Crop"))

    st.markdown("### Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.caption(
        "Charts summarize the current dataset. They do not independently "
        "validate model accuracy or guarantee farming outcomes."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------
st.divider()
st.caption(
    "CropWise AI | Academic Capstone Prototype | "
    "Python + Streamlit + TabPFN"
)
