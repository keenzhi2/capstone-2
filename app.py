
import streamlit as st
import pandas as pd
import numpy as np
import joblib
model = joblib.load("best_xgboost_model (1).pkl")  # Load model here
scaler = joblib.load("scaler (1).pkl")

st.title("Sovereign Debt Crisis Predictor")
st.write("Enter the details below to check for a sovereign debt crisis.")

country = st.text_input("Enter Country Name")
year = st.number_input("Enter Year", min_value=2000, max_value=2024, step=1)


access_to_electricity = st.number_input("Access to electricity (% of population)", min_value=0.0, max_value=100.0, step=0.1)
age_dependency_ratio = st.number_input("Age dependency ratio (% of working-age population)", min_value=0.0, step=0.1)
broad_money_to_reserves_ratio = st.number_input("Broad money to total reserves ratio", min_value=0.0, step=0.1)
claims_on_central_government = st.number_input("Claims on central government, etc. (% GDP)", min_value=0.0, step=0.1)
consumer_price_index = st.number_input("Consumer price index (2010 = 100)", min_value=0.0, step=0.1)
control_of_corruption = st.number_input("Control of Corruption: Estimate", min_value=0.0, max_value=100.0, step=0.1)
current_account_balance = st.number_input("Current account balance (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
current_health_expenditure = st.number_input("Current health expenditure (% of GDP)", min_value=0.0, step=0.1)
domestic_credit_private_sector = st.number_input("Domestic credit to private sector (% of GDP)", min_value=0.0, step=0.1)
exports_of_goods_services = st.number_input("Exports of goods and services (current US$)", min_value=0.0, step=1.0)
external_balance_on_goods_services = st.number_input("External balance on goods and services (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
foreign_direct_investment_inflows = st.number_input("Foreign direct investment, net inflows (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
foreign_direct_investment_outflows = st.number_input("Foreign direct investment, net outflows (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
gdp_growth = st.number_input("GDP growth (annual %)", min_value=-100.0, max_value=100.0, step=0.1)
gov_final_consumption_expenditure = st.number_input("General government final consumption expenditure (% of GDP)", min_value=0.0, step=0.1)
government_effectiveness = st.number_input("Government Effectiveness: Estimate", min_value=0.0, max_value=100.0, step=0.1)
gross_savings = st.number_input("Gross savings (% of GDP)", min_value=0.0, step=0.1)
imports_of_goods_services = st.number_input("Imports of goods and services (current US$)", min_value=0.0, step=1.0)
inflation_consumer_prices = st.number_input("Inflation, consumer prices (annual %)", min_value=-100.0, max_value=100.0, step=0.1)
labor_force_participation_rate = st.number_input("Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)", min_value=0.0, max_value=100.0, step=0.1)
official_exchange_rate = st.number_input("Official exchange rate (LCU per US$, period average)", min_value=0.0, step=0.1)
basic_drinking_water_services = st.number_input("People using at least basic drinking water services (% of population)", min_value=0.0, max_value=100.0, step=0.1)
political_stability = st.number_input("Political Stability and Absence of Violence/Terrorism: Estimate", min_value=0.0, max_value=100.0, step=0.1)
polity_combined_polity_score = st.number_input("Polity database: Combined Polity Score", min_value=-10, max_value=10, step=1)
polity_regime_durability = st.number_input("Polity database: Regime Durability Index", min_value=0, step=1)
state_fragility_index = st.number_input("State Fragility Index", min_value=0, step=1)
total_reserves_minus_gold = st.number_input("Total reserves minus gold (current US$)", min_value=0.0, step=1.0)
use_of_imf_credit = st.number_input("Use of IMF credit (DOD, current US$)", min_value=0.0, step=1.0)

# Prepare input data in the same format as model expects
input_data = pd.DataFrame({
    'Access to electricity (% of population)': [access_to_electricity],
    'Age dependency ratio (% of working-age population)': [age_dependency_ratio],
    'Broad money to total reserves ratio': [broad_money_to_reserves_ratio],
    'Claims on central government, etc. (% GDP)': [claims_on_central_government],
    'Consumer price index (2010 = 100)': [consumer_price_index],
    'Control of Corruption: Estimate': [control_of_corruption],
    'Current account balance (% of GDP)': [current_account_balance],
    'Current health expenditure (% of GDP)': [current_health_expenditure],
    'Domestic credit to private sector (% of GDP)': [domestic_credit_private_sector],
    'Exports of goods and services (current US$)': [exports_of_goods_services],
    'External balance on goods and services (% of GDP)': [external_balance_on_goods_services],
    'Foreign direct investment, net inflows (% of GDP)': [foreign_direct_investment_inflows],
    'Foreign direct investment, net outflows (% of GDP)': [foreign_direct_investment_outflows],
    'GDP growth (annual %)': [gdp_growth],
    'General government final consumption expenditure (% of GDP)': [gov_final_consumption_expenditure],
    'Government Effectiveness: Estimate': [government_effectiveness],
    'Gross savings (% of GDP)': [gross_savings],
    'Imports of goods and services (current US$)': [imports_of_goods_services],
    'Inflation, consumer prices (annual %)': [inflation_consumer_prices],
    'Labor force participation rate, total (% of total population ages 15+) (modeled ILO estimate)': [labor_force_participation_rate],
    'Official exchange rate (LCU per US$, period average)': [official_exchange_rate],
    'People using at least basic drinking water services (% of population)': [basic_drinking_water_services],
    'Political Stability and Absence of Violence/Terrorism: Estimate': [political_stability],
    'Polity database: Combined Polity Score': [polity_combined_polity_score],
    'Polity database: Regime Durability Index': [polity_regime_durability],
    'State Fragility Index': [state_fragility_index],
    'Total reserves minus gold (current US$)': [total_reserves_minus_gold],
    'Use of IMF credit (DOD, current US$)': [use_of_imf_credit]
})

# When user clicks the predict button
if st.button("🔍 Predict Crisis"):
    # Check if all inputs are provided
    if input_data.isnull().any().any():
      st.warning("Please provide all input values!")
    else:
        # Scale the input data using the same scaler that was used during training
        input_data_scaled = scaler.transform(input_data)

        # Make prediction using the trained model
        prediction = model.predict(input_data_scaled)[0]
        result = "🔴 Crisis" if prediction == 1 else "🟢 No Crisis"

        # Display the result
        st.subheader(f"Prediction for {country} in {year}:")
        st.markdown(f"## {result}")

        # Optionally, show the feature values that contributed to the prediction
        st.write("### Input Features:")
        for feature, value in input_data.iloc[0].items():
            st.write(f"{feature}: {value}")
