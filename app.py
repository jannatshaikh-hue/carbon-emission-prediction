
import streamlit as st
import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
import numpy as np

# --- Start of fix: Create a dummy model if it doesn't exist ---
model_filename = "carbon_model.pkl"

try:
    # Try to load the model to check if it exists
    model = joblib.load(model_filename)
except FileNotFoundError:
    # If not found, create and save a dummy model
    st.warning(f"'{model_filename}' not found. Creating a dummy model for demonstration.")

    # Define dummy data for training
    X_dummy = pd.DataFrame({
        'distance_km': np.random.rand(100) * 100,
        'fuel_consumption': np.random.rand(100) * 10,
        'transport_mode_Bus': np.random.randint(0, 2, 100),
        'transport_mode_Car': np.random.randint(0, 2, 100),
        'transport_mode_Train': np.random.randint(0, 2, 100)
    })
    # Simple dummy target (e.g., CO2 emission = 0.2 * distance + 2.5 * fuel)
    y_dummy = 0.2 * X_dummy['distance_km'] + 2.5 * X_dummy['fuel_consumption'] + np.random.rand(100) * 5

    # Train a simple linear regression model
    dummy_model = LinearRegression()
    dummy_model.fit(X_dummy, y_dummy)

    # Save the dummy model
    joblib.dump(dummy_model, model_filename)
    model = dummy_model # Assign the newly created model
# --- End of fix ---


st.title("🚗 Daily Carbon Emission Predictor")

st.write("Enter your daily travel details:")

# Input fields
distance_km = st.number_input("Distance Travelled (km)", min_value=0.0)
fuel_consumption = st.number_input("Fuel Consumption (liters)", min_value=0.0)

transport_mode = st.selectbox(
    "Transport Mode",
    ["Bike", "Bus", "Car", "Train"]
)

# One-hot encoding
data = {
    "distance_km": distance_km,
    "fuel_consumption": fuel_consumption,
    "transport_mode_Bus": 0,
    "transport_mode_Car": 0,
    "transport_mode_Train": 0
}

if transport_mode != "Bike":
    data[f"transport_mode_{transport_mode}"] = 1

input_df = pd.DataFrame([data])

# Predict
if st.button("Predict CO₂ Emission"):
    # Ensure the input_df columns match the model's expected features
    # The dummy model expects: distance_km, fuel_consumption, transport_mode_Bus, transport_mode_Car, transport_mode_Train
    expected_features = ['distance_km', 'fuel_consumption', 'transport_mode_Bus', 'transport_mode_Car', 'transport_mode_Train']
    input_df_reordered = input_df[expected_features]

    prediction = model.predict(input_df_reordered)
    st.success(f"🌱 Estimated CO₂ Emission: {prediction[0]:.2f} kg/day")
