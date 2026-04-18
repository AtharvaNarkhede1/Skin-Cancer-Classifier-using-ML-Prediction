import streamlit as st
import requests
from PIL import Image
from io import BytesIO
import os

st.set_page_config(
    page_title="Skin Cancer Detection",
    page_icon="🔬",
    layout="centered"
)

# API endpoint (FastAPI)
API_URL = "http://localhost:8000/predict"

st.title("🔬 Skin Lesion Classifier")
st.write("Upload a skin lesion image to detect if it's **Benign** or **Malignant**.")

# Custom CSS for aesthetics
st.markdown("""
<style>
    .result-card {
        padding: 20px;
        border-radius: 10px;
        margin-top: 20px;
        text-align: center;
    }
    .benign {
        background-color: #d4edda;
        color: #155724;
        border: 1px solid #c3e6cb;
    }
    .malignant {
        background-color: #f8d7da;
        color: #721c24;
        border: 1px solid #f5c6cb;
    }
    .confidence-text {
        font-size: 1.2rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # Display preview
    image = Image.open(uploaded_file)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    if st.button('Analyze Image'):
        with st.spinner('Processing...'):
            try:
                # Prepare file for requests
                files = {"file": (uploaded_file.name, uploaded_file.getvalue(), uploaded_file.type)}
                
                # Call API
                response = requests.post(API_URL, files=files)
                
                if response.status_code == 200:
                    result = response.json()
                    prediction = result["prediction"]
                    confidence = result["confidence"]
                    
                    # Display result
                    css_class = "malignant" if prediction == "Malignant" else "benign"
                    st.markdown(f"""
                    <div class="result-card {css_class}">
                        <h2>{prediction}</h2>
                        <p class="confidence-text">Confidence: {confidence*100:.2f}%</p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    if prediction == "Malignant":
                        st.warning("⚠️ High risk detected. Please consult a dermatologist immediately.")
                    else:
                        st.success("✅ This lesion appears to be benign.")
                        
                elif response.status_code == 503:
                    st.error("Model not loaded on server. Is the training complete?")
                else:
                    st.error(f"Error from API: {response.text}")
                    
            except requests.exceptions.ConnectionError:
                st.error("Could not connect to the Backend API. Make sure src/app.py is running on port 8000.")
            except Exception as e:
                st.error(f"An error occurred: {e}")

st.divider()
st.info("Disclaimer: This tool is for educational/demonstration purposes only and is not a substitute for professional medical advice, diagnosis, or treatment.")
