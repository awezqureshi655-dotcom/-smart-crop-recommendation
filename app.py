
import streamlit as st
import pandas as pd
import joblib

# Load model, encoder and dataset
model = joblib.load("crop_model.pkl")
encoder = joblib.load("crop_encoder.pkl")
df = pd.read_csv("crop_data.csv")

# Page settings
st.set_page_config(
    page_title="AI Smart Crop Recommendation",
    page_icon="🌱",
    layout="centered"
)

# Title
st.title("🌱 AI-Based Smart Crop Recommendation System")
st.write("Enter the soil and weather conditions to get AI-based crop recommendations.")

st.divider()

# Input section
st.subheader("Enter Farm Conditions")

col1, col2 = st.columns(2)

with col1:
    N = st.number_input("Nitrogen (N)", min_value=0.0, value=90.0)
    P = st.number_input("Phosphorus (P)", min_value=0.0, value=42.0)
    K = st.number_input("Potassium (K)", min_value=0.0, value=43.0)
    temperature = st.number_input("Temperature (°C)", value=20.88)

with col2:
    humidity = st.number_input("Humidity (%)", value=82.0)
    ph = st.number_input("Soil pH", min_value=0.0, max_value=14.0, value=6.5)
    rainfall = st.number_input("Rainfall (mm)", min_value=0.0, value=202.94)

st.divider()

# Recommendation button
if st.button("🌾 Recommend Crop", use_container_width=True):

    new_input = pd.DataFrame([{
        "N": N,
        "P": P,
        "K": K,
        "temperature": temperature,
        "humidity": humidity,
        "ph": ph,
        "rainfall": rainfall
    }])

    # Get probabilities
    probabilities = model.predict_proba(new_input)[0]

    # Top 3 crops
    top3_indices = probabilities.argsort()[-3:][::-1]

    st.success(
        f"Recommended Crop: {encoder.classes_[top3_indices[0]].upper()}"
    )

    st.subheader("🌱 Top 3 AI Recommendations")

    for rank, index in enumerate(top3_indices, start=1):
        crop = encoder.classes_[index]
        probability = probabilities[index] * 100

        st.write(
            f"**{rank}. {crop.upper()} — {probability:.2f}%**"
        )

    # Details of primary recommendation
    recommended_crop = encoder.classes_[top3_indices[0]]
    crop_info = df[df["label"] == recommended_crop].iloc[0]

    st.subheader("📋 Crop Information")

    st.write("**Season:**", crop_info["season"])
    st.write("**Soil Type:**", crop_info["soil_type"])
    st.write("**Water Requirement:**", crop_info["water_need"])
    st.write("**Market:**", crop_info["market"])
    st.write("**State:**", crop_info["state"])
    st.write("**Price per Quintal:** ₹", crop_info["price_per_quintal"])

st.divider()

st.caption(
    "Prototype developed using Machine Learning and a curated crop dataset."
)
