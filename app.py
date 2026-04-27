import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

# ✅ CORRECT Gemini import
import google.generativeai as genai

# ---------------- CONFIG ----------------
st.set_page_config(page_title="AI Fertilizer System", layout="centered")

# ---------------- GEMINI SETUP ----------------
try:
    genai.configure(api_key=st.secrets["GOOGLE_API_KEY"])
    model_gemini = genai.GenerativeModel("gemini-1.5-flash")
    gemini_available = True
except:
    gemini_available = False

# ---------------- TITLE ----------------
st.title("🌱 AI Fertilizer Recommendation System")
st.write("Predicts best fertilizer + gives smart farming advice")

# ---------------- INPUTS ----------------
temp = st.number_input("Temperature (°C)", 0.0)
humidity = st.number_input("Humidity (%)", 0.0)
moisture = st.number_input("Moisture (%)", 0.0)

soil = st.selectbox("Soil Type", ["Black", "Clayey", "Loamy", "Sandy"])
crop = st.selectbox("Crop Type", ["Barley", "Wheat", "Rice", "Sugarcane"])

nitrogen = st.number_input("Nitrogen", 0)
potassium = st.number_input("Potassium", 0)
phosphorous = st.number_input("Phosphorous", 0)

# ---------------- ENCODING ----------------
le_soil = LabelEncoder()
le_crop = LabelEncoder()

soil_encoded = le_soil.fit_transform([soil])[0]
crop_encoded = le_crop.fit_transform([crop])[0]

# ---------------- MODEL (Dummy trained) ----------------
X = np.array([
    [30, 60, 40, 0, 0, 20, 15, 10],
    [25, 50, 35, 1, 1, 10, 20, 5],
    [35, 70, 60, 2, 2, 30, 10, 20],
    [28, 65, 50, 3, 3, 25, 15, 15]
])

y = ["10-26-26", "20-20-20", "Urea", "DAP"]

model = RandomForestClassifier()
model.fit(X, y)

# ---------------- PREDICT ----------------
if st.button("Predict"):

    input_data = np.array([[temp, humidity, moisture,
                            soil_encoded, crop_encoded,
                            nitrogen, potassium, phosphorous]])

    prediction = model.predict(input_data)[0]

    st.success(f"🌾 Recommended Fertilizer: {prediction}")

    # ---------------- GRAPH ----------------
    st.subheader("📊 Nutrient Analysis")

    nutrients = ["Nitrogen", "Phosphorous", "Potassium"]
    values = [nitrogen, phosphorous, potassium]

    fig, ax = plt.subplots()
    ax.bar(nutrients, values)
    ax.set_ylabel("Value")
    ax.set_title("Soil Nutrients")

    st.pyplot(fig)

    # ---------------- SOIL HEALTH METER ----------------
    st.subheader("🌱 Soil Health Meter")

    health_score = (nitrogen + potassium + phosphorous) / 3

    if health_score < 20:
        status = "Poor 🔴"
    elif health_score < 50:
        status = "Moderate 🟡"
    else:
        status = "Healthy 🟢"

    st.write(f"Health Score: {health_score:.2f}")
    st.write(f"Status: {status}")

    # ---------------- CONFIDENCE SCORE ----------------
    st.subheader("🧠 Confidence Score")

    confidence = model.predict_proba(input_data).max() * 100
    st.write(f"{confidence:.2f}% confident")

    # ---------------- AI ADVICE ----------------
    st.subheader("💡 Farming Advice")

    if gemini_available:
        try:
            prompt = f"""
            Give short farming advice for:
            Crop: {crop}
            Soil: {soil}
            Temperature: {temp}
            Moisture: {moisture}
            """

            response = model_gemini.generate_content(prompt)
            st.write(response.text)

        except Exception as e:
            st.warning("AI unavailable. Using smart offline advice.")

            st.write("""
            - Maintain proper irrigation  
            - Avoid over-fertilization  
            - Monitor crop health regularly  
            - Use organic compost when possible  
            """)

            st.error(str(e))  # shows real error (hackathon proof)

    else:
        st.warning("AI not configured. Using offline system.")

        st.write("""
        - Maintain proper irrigation  
        - Avoid over-fertilization  
        - Monitor crop health regularly  
        - Use organic compost when possible  
        """)

# ---------------- FOOTER ----------------
st.write("🚀 Made for AI Hackathon")