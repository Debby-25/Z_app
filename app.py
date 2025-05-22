import streamlit as st
import pandas as pd
import numpy as np
import joblib
from sklearn.preprocessing import StandardScaler

# Load trained model and scaler
model = joblib.load("wealth_model.pkl")  # Make sure you've saved the trained model in your directory
scaler = joblib.load("scaler.pkl")  # Load the scaler used for normalization

# Streamlit UI setup
st.title("Wealth Prediction Model for Africa")
st.write("Enter details to predict the wealth index of a location.")

# User inputs for all 15 features
ghsl_water_surface = st.number_input("Fraction of land classified as water surface", min_value=0.0, max_value=1.0, step=0.01)
ghsl_built_pre_1975 = st.number_input("Fraction of land built-up before 1975", min_value=0.0, max_value=1.0, step=0.01)
ghsl_built_1975_to_1990 = st.number_input("Fraction of land built-up from 1975 to 1990", min_value=0.0, max_value=1.0, step=0.01)
ghsl_built_1990_to_2000 = st.number_input("Fraction of land built-up from 1990 to 2000", min_value=0.0, max_value=1.0, step=0.01)
ghsl_built_2000_to_2014 = st.number_input("Fraction of land built-up from 2000 to 2014", min_value=0.0, max_value=1.0, step=0.01)
ghsl_not_built_up = st.number_input("Fraction of land never built-up", min_value=0.0, max_value=1.0, step=0.01)
ghsl_pop_density = st.number_input("Population density (within 5km radius)", min_value=0.0, step=0.1)
landcover_crops_fraction = st.number_input("Fraction of land classified as cropland", min_value=0.0, max_value=1.0, step=0.01)
landcover_urban_fraction = st.number_input("Fraction of land classified as urban", min_value=0.0, max_value=1.0, step=0.01)
landcover_water_permanent_10km_fraction = st.number_input("Fraction of land with permanent water (within 10km)", min_value=0.0, max_value=1.0, step=0.01)
landcover_water_seasonal_10km_fraction = st.number_input("Fraction of land with seasonal water (within 10km)", min_value=0.0, max_value=1.0, step=0.01)
nighttime_lights = st.number_input("Nighttime light intensity (economic activity indicator)", min_value=0.0, step=0.1)
dist_to_capital = st.number_input("Distance to the country's capital (km)", min_value=0.0, step=1.0)
dist_to_shoreline = st.number_input("Distance to the nearest ocean shoreline (km)", min_value=0.0, step=1.0)
urban_or_rural = st.selectbox("Is the cluster in an urban or rural setting?", ["U", "R"])

# Create feature array (ensuring ALL 15 features are included)
input_data = np.array([[ghsl_water_surface, ghsl_built_pre_1975, ghsl_built_1975_to_1990,
                        ghsl_built_1990_to_2000, ghsl_built_2000_to_2014, ghsl_not_built_up,
                        ghsl_pop_density, landcover_crops_fraction, landcover_urban_fraction,
                        landcover_water_permanent_10km_fraction, landcover_water_seasonal_10km_fraction,
                        nighttime_lights, dist_to_capital, dist_to_shoreline, 1 if urban_or_rural == "U" else 0]])  # Encoding urban/rural setting

# Apply scaling before prediction
input_scaled = scaler.transform(input_data)

# Predict wealth index
if st.button("Predict Wealth Index"):
    prediction = model.predict(input_scaled)
    st.write(f"Estimated Wealth Index: {prediction[0]:.4f}")