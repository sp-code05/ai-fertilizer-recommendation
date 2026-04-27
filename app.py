import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from google import genai

# ---------------- GEMINI CLIENT ----------------
client = genai.Client(api_key=st.secrets["GOOGLE_API_KEY"])

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

# ---------------- AI ADVICE ----------------
if st.session_state["prediction"] is not None:
    if st.button("💡 Get AI Advice"):

        prompt = f"""
A farmer has:

Soil Type: {soil_type}
Crop Type: {crop_type}
Temperature: {temperature}
Humidity: {humidity}
Nitrogen: {nitrogen}
Phosphorous: {phosphorous}
Potassium: {potassium}

Recommended fertilizer: {st.session_state["prediction"]}

Explain why this fertilizer is suitable and give simple farming advice.
"""

        try:
            response = client.models.generate_content(
                model="gemini-1.5-flash-latest",
                contents=prompt
            )

            st.success("🤖 AI Advice:")
            st.write(response.text)

        except Exception as e:
            st.error("REAL ERROR:")
            st.write(e)

            st.warning("⚠️ AI service unavailable. Showing fallback advice.")

            st.info(f"""
🌾 Recommended Fertilizer: {st.session_state["prediction"]}

✔ Based on trained agricultural dataset patterns.

💡 General Farming Advice:

- Maintain balanced irrigation
- Avoid excess nitrogen fertilizer
- Monitor crop health regularly
- Use organic compost when possible
""")

# ---------------- FOOTER ----------------
st.markdown("---")
st.markdown("Made for AI Hackathon 🚀")