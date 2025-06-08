
import time
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from io import BytesIO

# ----------------------------
# Safe rerun helper for Streamlit compatibility
# ----------------------------
def safe_rerun():
    try:
        st.experimental_rerun()
    except AttributeError:
        pass  # Older Streamlit version without rerun support

# ----------------------------
# Page Configuration
# ----------------------------
st.set_page_config(page_title="🛠️ Scrap Quantity Prediction App", layout="wide")

# ----------------------------
# Session States
# ----------------------------
if 'authenticated' not in st.session_state:
    st.session_state.authenticated = False

if 'logout' not in st.session_state:
    st.session_state.logout = False

if 'total_predictions' not in st.session_state:
    st.session_state.total_predictions = 0

if 'last_activity' not in st.session_state:
    st.session_state.last_activity = time.time()

if 'dark_mode' not in st.session_state:
    st.session_state.dark_mode = False

# ----------------------------
# Session Timeout
# ----------------------------
session_timeout = 30 * 60  # 30 minutes
if time.time() - st.session_state.last_activity > session_timeout:
    st.session_state.authenticated = False
    st.session_state.logout = True
    safe_rerun()

# ----------------------------
# Authentication
# ----------------------------
def login_page():
    st.title("🔐 Scrap Quantity Prediction Login")
    username = st.text_input("👤 Username")
    password = st.text_input("🔑 Password", type="password")
    if st.button("🔓 Login"):
        if username == "admin" and password == "password":
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.session_state.last_activity = time.time()
            safe_rerun()
        else:
            st.error("❌ Invalid username or password")

def logout():
    st.session_state.authenticated = False
    st.session_state.logout = True

# ----------------------------
# Dark Mode Styling
# ----------------------------
def set_dark_mode():
    if st.session_state.dark_mode:
        st.markdown("""
        <style>
        html, body, [class*="css"]  {
            background-color: #181818 !important;
            color: #ffffff !important;
        }
        .stButton > button {
            background-color: #333333;
            color: white;
        }
        .stTextInput input, .stSelectbox div, .stSlider > div {
            background-color: #2e2e2e;
            color: white;
        }
        .stDownloadButton > button {
            background-color: #444;
            color: white;
        }
        </style>
        """, unsafe_allow_html=True)

# ----------------------------
# Prediction Page
# ----------------------------
def prediction_app():
    st.title("📊 Scrap Quantity Prediction")

    # Load your model, scaler and columns
    with open("model.pkl", "rb") as f:
        model = pickle.load(f)
        
    with open("scaler.pkl", "rb") as f:
        scaler = pickle.load(f)
        
    with open("model_columns.pkl", "rb") as f:
        model_columns = pickle.load(f)

    st.subheader("📝 Input Work Order Details")

    # User inputs - all manual inputs
    with st.expander("🔧 Customize Input Features", expanded=True):
        OrderQty = st.number_input("Order Quantity", min_value=0, step=1, value=100, format="%d")
        ProductID = st.number_input("Product ID", min_value=0, step=1, value=123, format="%d")
        ScrapReasonID = st.number_input("Scrap Reason ID", min_value=0, step=1, value=1, format="%d")
        StockedQty = st.number_input("Stocked Quantity", min_value=0, step=1, value=80, format="%d")
        Weight = st.number_input("Weight", min_value=0.0, step=0.01, value=12.5, format="%.2f")
        WorkOrderID = st.number_input("Work Order ID", min_value=0, step=1, value=555, format="%d")
        Name_Paint_process_failed = st.selectbox("Paint process failed", [0,1], index=0)
        Name_Primer_process_failed = st.selectbox("Primer process failed", [0,1], index=0)
        Name_Thermoform_temperature_too_low = st.selectbox("Thermoform temperature too low", [0,1], index=0)
        Name_Trim_length_too_short = st.selectbox("Trim length too short", [0,1], index=0)

    input_data = {
        'OrderQty': OrderQty,
        'ProductID': ProductID,
        'ScrapReasonID': ScrapReasonID,
        'StockedQty': StockedQty,
        'Weight': Weight,
        'WorkOrderID': WorkOrderID,
        'Name_Paint process failed': Name_Paint_process_failed,
        'Name_Primer process failed': Name_Primer_process_failed,
        'Name_Thermoform temperature too low': Name_Thermoform_temperature_too_low,
        'Name_Trim length too short': Name_Trim_length_too_short,
    }

    input_df = pd.DataFrame([input_data])

    # Reorder columns to model columns and fill missing with 0 (if any)
    input_df = input_df.reindex(columns=model_columns, fill_value=0)

    # Scale inputs
    input_scaled = scaler.transform(input_df)

    # Predict scrap quantity
    prediction = model.predict(input_scaled)

    st.subheader("🔍 Prediction Result")
    st.write(f"🧮 Predicted Scrap Quantity: **{prediction[0]:.2f}**")

    # Optionally show input data and allow download
    if st.checkbox("Show Input Data"):
        st.write(input_df)

    csv = input_df.copy()
    csv['Predicted_ScrapQty'] = prediction
    csv_data = csv.to_csv(index=False).encode()

    st.download_button("📥 Download Prediction Report", data=csv_data, file_name="scrap_quantity_prediction.csv", mime="text/csv")

    st.session_state.total_predictions += 1
    st.session_state.last_activity = time.time()

# ----------------------------
# Sidebar Navigation
# ----------------------------
st.sidebar.title("🧭 Navigation")

if st.session_state.authenticated:
    st.sidebar.markdown("### 👋 Welcome, Admin!")
    st.sidebar.markdown(f"### 📊 Total Predictions Made: `{st.session_state.total_predictions}`")

    st.sidebar.subheader("🌓 Dark Mode")
    st.session_state.dark_mode = st.sidebar.checkbox("Enable Dark Mode")
    set_dark_mode()

    st.sidebar.subheader("🛠️ Help")
    st.sidebar.markdown("For more info, visit the [📄 documentation](https://example.com).")

    st.sidebar.subheader("⭐ Rate Your Experience")
    rating = st.sidebar.slider("Rate the app", 1, 5)
    if rating:
        st.sidebar.markdown(f"🙏 Thanks for rating us {rating}/5!")

    st.sidebar.button("🚪 Logout", on_click=logout)

    prediction_app()
else:
    login_page()

if st.session_state.logout:
    st.session_state.logout = False
    safe_rerun()

# ----------------------------
# Footer
# ----------------------------
st.markdown("---")
st.markdown("<center>Made with ❤️ by Somya Khare</center>", unsafe_allow_html=True)
