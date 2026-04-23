import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
import pickle

# Load dataset
data = pd.read_csv("fertilizer_data.csv")

# Clean column names (remove spaces)
data.columns = data.columns.str.strip()

# Print columns (for checking)
print("Columns:", data.columns)

# Encode categorical columns
le_soil = LabelEncoder()
le_crop = LabelEncoder()

data["Soil Type"] = le_soil.fit_transform(data["Soil Type"])
data["Crop Type"] = le_crop.fit_transform(data["Crop Type"])

# Define input (X) and output (y)
X = data.drop("Fertilizer Name", axis=1)
y = data["Fertilizer Name"]

# Create model
model = RandomForestClassifier()

# Train model
model.fit(X, y)

# Save model and encoders
pickle.dump(model, open("model.pkl", "wb"))
pickle.dump(le_soil, open("soil_encoder.pkl", "wb"))
pickle.dump(le_crop, open("crop_encoder.pkl", "wb"))

print("✅ Model trained successfully!")