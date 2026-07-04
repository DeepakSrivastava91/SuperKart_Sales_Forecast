import os
import joblib
import pandas as pd
import streamlit as st
from huggingface_hub import hf_hub_download

HF_USERNAME = os.getenv("HF_USERNAME", "your-hf-username")
MODEL_REPO = os.getenv("MODEL_REPO", f"{HF_USERNAME}/superkart-sales-model")
REFERENCE_YEAR = 2009

# Download and load the model pipeline from the HF model hub
model_path = hf_hub_download(repo_id=MODEL_REPO, filename="best_superkart_sales_model_v1.joblib")
model = joblib.load(model_path)

st.title("SuperKart Sales Forecast")
st.write("Enter the product and store details to predict the total product-store sales revenue.")

col1, col2 = st.columns(2)
with col1:
    Product_Weight = st.number_input("Product Weight", 1.0, 50.0, 12.66)
    Product_Sugar_Content = st.selectbox("Product Sugar Content", ["Low Sugar", "Regular", "No Sugar"])
    Product_Allocated_Area = st.number_input("Product Allocated Area (ratio)", 0.0, 1.0, 0.05, format="%.3f")
    Product_Type = st.selectbox("Product Type", [
        "Baking Goods", "Breads", "Breakfast", "Canned", "Dairy", "Frozen Foods",
        "Fruits and Vegetables", "Hard Drinks", "Health and Hygiene", "Household",
        "Meat", "Others", "Seafood", "Snack Foods", "Soft Drinks", "Starchy Foods"])
    Product_MRP = st.number_input("Product MRP", 1.0, 500.0, 150.0)
    Product_Category = st.selectbox("Product Category (Id prefix)", ["FD", "NC", "DR"])
with col2:
    Store_Id = st.selectbox("Store Id", ["OUT001", "OUT002", "OUT003", "OUT004"])
    Store_Establishment_Year = st.number_input("Store Establishment Year", 1980, REFERENCE_YEAR, 1999)
    Store_Size = st.selectbox("Store Size", ["Small", "Medium", "High"])
    Store_Location_City_Type = st.selectbox("Store Location City Type", ["Tier 1", "Tier 2", "Tier 3"])
    Store_Type = st.selectbox("Store Type", [
        "Departmental Store", "Food Mart", "Supermarket Type1", "Supermarket Type2"])

input_data = pd.DataFrame([{
    "Product_Weight": Product_Weight,
    "Product_Allocated_Area": Product_Allocated_Area,
    "Product_MRP": Product_MRP,
    "Store_Age": REFERENCE_YEAR - Store_Establishment_Year,
    "Product_Sugar_Content": Product_Sugar_Content,
    "Product_Type": Product_Type,
    "Store_Id": Store_Id,
    "Store_Size": Store_Size,
    "Store_Location_City_Type": Store_Location_City_Type,
    "Store_Type": Store_Type,
    "Product_Category": Product_Category,
}])

if st.button("Predict Sales"):
    prediction = model.predict(input_data)[0]
    st.success(f"Predicted Product Store Sales Total: {prediction:,.2f}")
