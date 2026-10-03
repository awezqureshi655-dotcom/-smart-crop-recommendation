import streamlit as st
import pandas as pd
import os
from tabpfn_client import TabPFNClassifier

st.set_page_config(
    page_title="AI Smart Crop Recommendation",
    page_icon="🌱"
)

# TabPFN authentication
os.environ["TABPFN_TOKEN"] = st.secrets["TABPFN_TOKEN"]

# Load dataset
df = pd.read_csv("crop_data.csv")

features = [
    "N",
    "P",
    "K",
    "temperature",
    "humidity",
    "ph",
    "rainfall"
]

st.title("🌱 AI-Based Smart Crop Recommendation System")

st.write(
    "Enter the soil and weather conditions to get an "
    "AI-based crop recommendation using TabPFN."
)

st.divider()

st.subheader("🌾 Soil & Weather Conditions")

N = st.number_input("Nitrogen (N)", min_value=0.0, max_value=200.0, value=90.0)
P = st.number_input("Phosphorus (P)", min_value=0.0, max_value=200.0, value=42.0)
K = st.number_input("Potassium (K)", min_value=0.0, max_value=200.0, value=43.0)

temperature = st.number_input(
    "Temperature (°C)",
    min_value=-10.0,
    max_value=60.0,
    value=20.88
)

humidity = st.number_input(
    "Humidity (%)",
    min_value=0.0,
    max_value=100.0,
    value=82.0
)

ph = st.number_input(
    "Soil pH",
    min_value=0.0,
    max_value=14.0,
    value=6.5
)

rainfall = st.number_input(
    "Rainfall (mm)",
    min_value=0.0,
    max_value=1000.0,
    value=202.94
)

st.divider()

if st.button("🌱 Recommend Crop", use_container_width=True):

    with st.spinner("TabPFN AI is analyzing the conditions..."):

        try:
            X = df[features]
            y = df["label"]

            model = TabPFNClassifier()

            model.fit(X, y)

            new_input = pd.DataFrame([{
                "N": N,
                "P": P,
                "K": K,
                "temperature": temperature,
                "humidity": humidity,
                "ph": ph,
                "rainfall": rainfall
            }])

            prediction = model.predict(new_input)[0]

            probabilities = model.predict_proba(new_input)[0]
            classes = model.classes_

            top3_indices = probabilities.argsort()[-3:][::-1]

            st.success(
                f"🌱 Recommended Crop: {prediction.upper()}"
            )

            st.subheader("📊 Top 3 AI Recommendations")

            for rank, index in enumerate(top3_indices, start=1):

                crop = classes[index]
                probability = probabilities[index] * 100

                st.write(
                    f"**{rank}. {crop.upper()} — "
                    f"{probability:.2f}% model confidence**"
                )

                st.progress(
                    min(float(probability) / 100, 1.0)
                )

            st.divider()

            crop_rows = df[
                df["label"].str.lower() == str(prediction).lower()
            ]

            if not crop_rows.empty:

                crop_info = crop_rows.iloc[0]

                st.subheader("🌾 Crop Information")

                col1, col2 = st.columns(2)

                with col1:
                    st.write(f"**Season:** {crop_info['season']}")
                    st.write(f"**Soil Type:** {crop_info['soil_type']}")

                with col2:
                    st.write(f"**Water Need:** {crop_info['water_need']}")
                    st.write(f"**State:** {crop_info['state']}")

            st.caption(
                "Note: The displayed percentages are TabPFN model "
                "confidence scores and are not guaranteed real-world "
                "crop success probabilities."
            )

        except Exception as e:

            st.error("The TabPFN model could not run.")
            st.code(str(e))
