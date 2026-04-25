import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import google.generativeai as genai

# 🔑 Put your API key here (for local testing)
genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])

# ---------------- MODEL ----------------
@st.cache_resource
def load_model():
    data = pd.read_csv("fertilizer_data.csv")
    data.columns = data.columns.str.strip()

    le_soil = LabelEncoder()
    le_crop = LabelEncoder()

    data["Soil Type"] = le_soil.fit_transform(data["Soil Type"])
    data["Crop Type"] = le_crop.fit_transform(data["Crop Type"])

    X = data.drop("Fertilizer Name", axis=1)
    y = data["Fertilizer Name"]

    model = RandomForestClassifier()
    model.fit(X, y)

    return model, le_soil, le_crop

model, soil_encoder, crop_encoder = load_model()

# ---------------- UI ----------------
st.title("🌱 AI Fertilizer Recommendation System")

st.markdown("""
This system predicts the best fertilizer and also gives AI-based farming advice using Google AI.
""")

st.info("👉 Enter all values and click Predict")

# Inputs
temperature = st.number_input("Temperature (°C)", 0.0, 50.0)
humidity = st.number_input("Humidity (%)", 0.0, 100.0)
moisture = st.number_input("Moisture (%)", 0.0, 100.0)

soil_type = st.selectbox("Soil Type", soil_encoder.classes_)
crop_type = st.selectbox("Crop Type", crop_encoder.classes_)

nitrogen = st.number_input("Nitrogen", 0, 100)
potassium = st.number_input("Potassium", 0, 100)
phosphorous = st.number_input("Phosphorous", 0, 100)

# Session state
if "prediction" not in st.session_state:
    st.session_state["prediction"] = None

# ---------------- PREDICTION ----------------
if st.button("🔍 Predict Fertilizer"):
    soil_encoded = soil_encoder.transform([soil_type])[0]
    crop_encoded = crop_encoder.transform([crop_type])[0]

    input_data = np.array([[temperature, humidity, moisture,
                            soil_encoded, crop_encoded,
                            nitrogen, potassium, phosphorous]])

    prediction = model.predict(input_data)
    st.session_state["prediction"] = prediction[0]

    st.success(f"🌾 Recommended Fertilizer: {prediction[0]}")

# ---------------- AI ADVICE (FINAL WORKING) ----------------
if st.session_state["prediction"] is not None:
    if st.button("💡 Get AI Advice"):

        prompt = f"""
        A farmer has:
        Soil Type: {soil_type}
        Crop Type: {crop_type}
        Temperature: {temperature}
        Humidity: {humidity}
        Nitrogen: {nitrogen}

        Recommended fertilizer: {st.session_state["prediction"]}

        Explain why this fertilizer is suitable and give simple farming advice.
        """

        try:
            model_gemini = genai.GenerativeModel("gemini-1.5-flash-latest")
            response = model_gemini.generate_content(prompt)

            st.write("🤖 AI Advice:")
            st.write(response.text)
            st.write("Note: Prediction is based on training data patterns.")

        except Exception as e:
            # 🔥 fallback (VERY IMPORTANT for hackathon)
            st.warning("AI unavailable, showing basic advice")

            st.write(f"""
            🌾 Recommended Fertilizer: **{st.session_state["prediction"]}**

            This fertilizer is suitable based on your soil and nutrient levels.
            Ensure proper irrigation and monitor crop health regularly.
            """)
# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made for AI Hackathon 🚀")