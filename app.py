
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
.main { background-color: #f7faf7; }
h1, h2, h3 { color: #205c39; }
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
# AUTHENTICATION
# --------------------------------------------------
try:
    os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]
except Exception:
    st.error(
        "TabPFN secret is missing. Add TABPFN_TOKEN "
        "in Streamlit app settings."
    )
    st.stop()

# --------------------------------------------------
# LOAD NEW MODEL DATASET
# --------------------------------------------------
try:
    df = pd.read_csv("crop_data_800.csv")
except Exception as e:
    st.error("Could not load crop_data_800.csv.")
    st.code(str(e))
    st.stop()

features = [
    "N", "P", "K", "temperature",
    "humidity", "ph", "rainfall"
]
required = features + ["label"]
missing = [c for c in required if c not in df.columns]

if missing:
    st.error(f"New dataset is missing columns: {missing}")
    st.stop()

for col in features:
    df[col] = pd.to_numeric(df[col], errors="coerce")

df["label"] = df["label"].astype(str).str.strip()
df = df.dropna(subset=features + ["label"])
df = df[df["label"] != ""]

if df.empty:
    st.error("The new dataset has no valid training records.")
    st.stop()

# --------------------------------------------------
# LOAD OLD DATASET AS OPTIONAL CROP INFORMATION
# --------------------------------------------------
metadata = pd.DataFrame()

try:
    old_df = pd.read_csv("crop_data.csv")
    if "label" in old_df.columns:
        old_df["label"] = old_df["label"].astype(str).str.strip()
        metadata = old_df
except Exception:
    pass

# --------------------------------------------------
# HEADER AND METRICS
# --------------------------------------------------
st.title("🌱 CropWise AI")
st.write("Smart crop recommendations and agricultural insights")
st.caption(
    "Academic capstone prototype | Python + Streamlit + TabPFN"
)

m1, m2, m3 = st.columns(3)
m1.metric("AI Training Records", len(df))
m2.metric("Crop Categories", df["label"].nunique())
m3.metric("Input Features", len(features))

st.caption(
    "The model uses crop_data_800.csv. "
    "Stored information from the original dataset is shown separately."
)
st.divider()

# --------------------------------------------------
# SIX DASHBOARD TABS
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
    st.write("Enter soil nutrient and weather measurements.")

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
            with st.spinner("TabPFN is analyzing your inputs..."):
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

        crop_rows = metadata[
            metadata["label"].str.lower() == prediction.lower()
        ] if "label" in metadata.columns else pd.DataFrame()

        if not crop_rows.empty:
            row = crop_rows.iloc[0]
            st.subheader("Available Stored Crop Information")

            info1, info2 = st.columns(2)

            for col, label in [
                ("season", "Season"),
                ("soil_type", "Dataset soil type"),
                ("water_need", "Water need"),
                ("state", "Dataset state")
            ]:
                if col in row.index and pd.notna(row[col]):
                    target = info1 if col in ["season", "soil_type"] else info2
                    target.write(f"**{label}:** {row[col]}")

            if "price_per_quintal" in crop_rows.columns:
                prices = pd.to_numeric(
                    crop_rows["price_per_quintal"], errors="coerce"
                ).dropna()

                if not prices.empty:
                    st.metric(
                        "Average Stored Price",
                        f"₹{prices.mean():,.0f}/quintal"
                    )
                    st.caption(
                        "Stored dataset value only; not a verified live market price."
                    )
        else:
            st.info(
                "No matching crop details are available in the original dataset."
            )

        st.caption(
            "Model confidence is not the probability of crop success, "
            "yield or profit. Results depend on dataset quality."
        )
    else:
        st.info("Enter field conditions and tap Recommend Crop.")

# --------------------------------------------------
# SOIL HEALTH
# --------------------------------------------------
with soil_tab:
    st.subheader("Soil Health Overview")
    st.write("Measurements currently entered in the Crop AI tab.")

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
        "These are user-entered measurements, not a laboratory diagnosis. "
        "Confirm nutrient units and soil condition with a soil test."
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
        st.write(f"**Crop to investigate:** {prediction.upper()}")
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

    if (
        "label" in metadata.columns
        and "price_per_quintal" in metadata.columns
    ):
        market_data = metadata.copy()
        market_data["price_per_quintal"] = pd.to_numeric(
            market_data["price_per_quintal"], errors="coerce"
        )
        market = (
            market_data.dropna(subset=["price_per_quintal"])
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

        if not market.empty:
            display_market = market.copy()
            for col in market.columns[1:]:
                display_market[col] = display_market[col].map(
                    lambda x: f"₹{x:,.0f}"
                )

            st.dataframe(
                display_market, use_container_width=True, hide_index=True
            )
            st.bar_chart(
                market.set_index("Crop")[["Average Price (₹/quintal)"]]
            )
        else:
            st.info("No stored price values are available.")
    else:
        st.info(
            "Market prices are unavailable. The new model dataset does not "
            "contain price information."
        )

    st.caption(
        "Any displayed prices are stored values from the original dataset, "
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
    a1.metric("Valid Training Records", len(df))
    a2.metric("Unique Crop Categories", df["label"].nunique())

    st.markdown("### Records by Crop")
    st.bar_chart(counts.set_index("Crop"))

    st.markdown("### New Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)

    st.caption(
        "Charts summarize the training dataset. They do not independently "
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
