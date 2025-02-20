
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

access_to_electricity = st.number_input("Access to Electricity (% of Population)", min_value=0.0, max_value=100.0, step=0.1)
age_dependency_ratio = st.number_input("Age Dependency Ratio (% of Working-Age Population)", min_value=0.0, step=0.1)
broad_money_to_reserves_ratio = st.number_input("Broad Money to Total Reserves Ratio", min_value=0.0, step=0.1)
claims_on_central_government = st.number_input("Claims on Central Government (% of GDP)", min_value=0.0, step=0.1)
consumer_price_index = st.number_input("Consumer Price Index (2010 = 100)", min_value=0.0, step=0.1)
control_of_corruption = st.number_input("Control of Corruption (Estimate)", min_value=0.0, max_value=100.0, step=0.1)
current_account_balance = st.number_input("Current Account Balance (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
current_health_expenditure = st.number_input("Current Health Expenditure (% of GDP)", min_value=0.0, step=0.1)
domestic_credit_private_sector = st.number_input("Domestic Credit to Private Sector (% of GDP)", min_value=0.0, step=0.1)
exports_of_goods_services = st.number_input("Exports of Goods and Services (current US$)", min_value=0.0, step=1.0)
external_balance_on_goods_services = st.number_input("External Balance on Goods and Services (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
foreign_direct_investment_inflows = st.number_input("Foreign Direct Investment Net Inflows (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
foreign_direct_investment_outflows = st.number_input("Foreign Direct Investment Net Outflows (% of GDP)", min_value=-100.0, max_value=100.0, step=0.1)
gdp_growth = st.number_input("GDP Growth (Annual %)", min_value=-100.0, max_value=100.0, step=0.1)
gov_final_consumption_expenditure = st.number_input("General Government Final Consumption Expenditure (% of GDP)", min_value=0.0, step=0.1)
gov_gross_debt_percent_of_gdp = st.number_input("General Government Gross Debt (% of GDP)", min_value=0.0, step=0.1)
government_effectiveness = st.number_input("Government Effectiveness (Estimate)", min_value=0.0, max_value=100.0, step=0.1)
gross_savings = st.number_input("Gross Savings (% of GDP)", min_value=0.0, step=0.1)
imports_of_goods_services = st.number_input("Imports of Goods and Services (current US$)", min_value=0.0, step=1.0)
inflation_consumer_prices = st.number_input("Inflation, Consumer Prices (Annual %)", min_value=-100.0, max_value=100.0, step=0.1)
labor_force_participation_rate = st.number_input("Labor Force Participation Rate (% of Total Population ages 15+)", min_value=0.0, max_value=100.0, step=0.1)
official_exchange_rate = st.number_input("Official Exchange Rate (LCU per US$, Period Average)", min_value=0.0, step=0.1)
basic_drinking_water_services = st.number_input("People Using at Least Basic Drinking Water Services (% of Population)", min_value=0.0, max_value=100.0, step=0.1)
political_stability = st.number_input("Political Stability and Absence of Violence/Terrorism (Estimate)", min_value=0.0, max_value=100.0, step=0.1)
polity_combined_polity_score = st.number_input("Polity Database: Combined Polity Score", min_value=-10, max_value=10, step=1)
polity_regime_durability = st.number_input("Polity Database: Regime Durability Index", min_value=0, step=1)
state_fragility_index = st.number_input("State Fragility Index", min_value=0, step=1)
total_reserves_minus_gold = st.number_input("Total Reserves Minus Gold (current US$)", min_value=0.0, step=1.0)
use_of_imf_credit = st.number_input("Use of IMF Credit (DOD, current US$)", min_value=0.0, step=1.0)

# Prepare input data in the same format as model expects
input_data = pd.DataFrame({
    'Access_to_Electricity': [access_to_electricity],
    'Age_Dependency_Ratio': [age_dependency_ratio],
    'Broad_Money_to_Reserves_Ratio': [broad_money_to_reserves_ratio],
    'Claims_on_Central_Government': [claims_on_central_government],
    'Consumer_Price_Index': [consumer_price_index],
    'Control_of_Corruption': [control_of_corruption],
    'Current_Account_Balance': [current_account_balance],
    'Current_Health_Expenditure': [current_health_expenditure],
    'Domestic_Credit_to_Private_Sector': [domestic_credit_private_sector],
    'Exports_of_Goods_Services': [exports_of_goods_services],
    'External_Balance_on_Goods_Services': [external_balance_on_goods_services],
    'Foreign_Direct_Investment_Net_Inflows': [foreign_direct_investment_inflows],
    'Foreign_Direct_Investment_Net_Outflows': [foreign_direct_investment_outflows],
    'GDP_Growth': [gdp_growth],
    'Gov_Final_Consumption_Expenditure': [gov_final_consumption_expenditure],
    'Gov_Gross_Debt_Percent_of_GDP': [gov_gross_debt_percent_of_gdp],
    'Government_Effectiveness': [government_effectiveness],
    'Gross_Savings': [gross_savings],
    'Imports_of_Goods_Services': [imports_of_goods_services],
    'Inflation_Consumer_Prices': [inflation_consumer_prices],
    'Labor_Force_Participation_Rate': [labor_force_participation_rate],
    'Official_Exchange_Rate': [official_exchange_rate],
    'Basic_Drinking_Water_Services': [basic_drinking_water_services],
    'Political_Stability': [political_stability],
    'Polity_Combined_Polity_Score': [polity_combined_polity_score],
    'Polity_Regime_Durability': [polity_regime_durability],
    'State_Fragility_Index': [state_fragility_index],
    'Total_Reserves_Minus_Gold': [total_reserves_minus_gold],
    'Use_of_IMF_Credit': [use_of_imf_credit]
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
