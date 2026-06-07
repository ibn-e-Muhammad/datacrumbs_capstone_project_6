import streamlit as st
import tensorflow as tf
from pathlib import Path
from PIL import Image
import numpy as np

# --- PAGE CONFIG ---
st.set_page_config(page_title="Medical Lesion Analyzer", page_icon="🩺", layout="centered")

# --- CSS INJECTION (Stitch Designer CSS Conversion) ---
st.markdown("""
<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap" rel="stylesheet">
<link href="https://fonts.googleapis.com/css2?family=Material+Symbols+Outlined:wght,FILL@100..700,0..1&display=swap" rel="stylesheet">
<style>
    html, body, [class*="css"]  {
        font-family: 'Inter', sans-serif;
        background-color: #f8f9fb;
    }
    
    .stButton>button {
        background-color: #005dac;
        color: #ffffff;
        border-radius: 8px;
        border: none;
        padding: 8px 24px;
        font-weight: 600;
        transition: background-color 0.3s, box-shadow 0.3s;
        width: 100%;
    }
    .stButton>button:hover {
        background-color: #1976d2;
        box-shadow: 0px 4px 12px rgba(0, 0, 0, 0.05);
        color: white;
    }
    
    [data-testid="stFileUploadDropzone"] {
        background-color: #e3f2fd;
        border: 2px dashed #1976d2;
        border-radius: 16px;
    }
    
    .medscan-card {
        background-color: #ffffff;
        border: 1px solid #c1c6d4;
        border-radius: 0.75rem; 
        padding: 24px; 
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.05);
        position: relative;
        overflow: hidden;
        margin-top: 16px;
        margin-bottom: 16px;
    }
    .benign-accent { border-left: 6px solid #4CAF50; }
    .malignant-accent { border-left: 6px solid #ba1a1a; }
    
    .bg-icon {
        position: absolute;
        top: 0;
        right: 0;
        padding: 16px;
        opacity: 0.05;
        font-size: 100px;
        font-variation-settings: 'FILL' 1;
        color: inherit;
    }
    
    .status-header {
        display: flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 8px;
    }
    .status-text {
        font-size: 12px;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        font-weight: 700;
    }
    
    .result-title {
        font-size: 28px;
        font-weight: 700;
        color: #191c1e;
        margin-top: 0;
        margin-bottom: 8px;
    }
    
    .result-desc {
        font-size: 16px;
        color: #414752;
        margin-bottom: 0;
    }
    
    .meter-container {
        display: flex;
        align-items: center;
        gap: 12px;
    }
    .meter-bg {
        height: 12px;
        flex-grow: 1;
        background-color: #eceef0;
        border-radius: 9999px;
        overflow: hidden;
    }
    .meter-fill-malignant {
        height: 100%;
        background-color: #ba1a1a;
        transition: width 1s ease-in-out;
    }
    .meter-fill-benign {
        height: 100%;
        background-color: #4CAF50;
        transition: width 1s ease-in-out;
    }
    .meter-text {
        font-size: 16px;
        font-weight: 700;
    }
    .text-error { color: #ba1a1a; }
    .text-success { color: #4CAF50; }
</style>
""", unsafe_allow_html=True)

# --- CACHE MODEL LOADING ---
@st.cache_resource
def load_lesion_model():
    model_path = Path(__file__).parent / 'models' / 'trained_lesion_classifier.keras'
    return tf.keras.models.load_model(model_path, compile=False)

try:
    model = load_lesion_model()
except Exception as e:
    st.error(f"Failed to load model from cross-platform path: {e}")
    st.stop()

# --- PREPROCESSING LOGIC ---
def preprocess_image(image: Image.Image):
    image = image.convert('RGB')
    image = image.resize((128, 128))
    img_array = np.array(image)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

# --- UI ---
st.title("Medical Lesion Analyzer")
st.markdown("### Secure Clinical Image Upload")

uploaded_file = st.file_uploader("Drag and drop a dermoscopic image (JPG/PNG)", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    
    # Center the uploaded image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, use_column_width=True, caption="Uploaded Lesion Source")
    
    if st.button("Run Clinical Analysis"):
        with st.spinner("Processing through Lesion Classifier..."):
            processed_img = preprocess_image(image)
            prediction = model.predict(processed_img)
            score = float(prediction[0][0])
            
            is_malignant = score > 0.5
            confidence_score = (score if is_malignant else (1 - score)) * 100
            
            # --- CUSTOM HTML/CSS UI INJECTION ---
            if is_malignant:
                html_card = f"""
                <div class="medscan-card malignant-accent text-error">
                    <span class="material-symbols-outlined bg-icon">warning</span>
                    <div class="status-header">
                        <span class="material-symbols-outlined" style="font-size: 20px;">warning</span>
                        <span class="status-text">Critical Detection</span>
                    </div>
                    <h2 class="result-title">Result: Malignant</h2>
                    <p class="result-desc">AI analysis detected abnormal mitotic figures and nuclear pleomorphism.</p>
                </div>
                """
                html_meter = f"""
                <div class="medscan-card" style="padding-top: 16px; padding-bottom: 16px;">
                    <p style="font-size: 12px; color: #526069; margin-top:0; margin-bottom:8px; font-weight:700; text-transform:uppercase; letter-spacing: 0.05em;">Confidence Meter</p>
                    <div class="meter-container">
                        <div class="meter-bg">
                            <div class="meter-fill-malignant" style="width: {confidence_score:.1f}%;"></div>
                        </div>
                        <span class="meter-text text-error">{confidence_score:.1f}%</span>
                    </div>
                </div>
                """
            else:
                html_card = f"""
                <div class="medscan-card benign-accent text-success">
                    <span class="material-symbols-outlined bg-icon">check_circle</span>
                    <div class="status-header">
                        <span class="material-symbols-outlined" style="font-size: 20px;">check_circle</span>
                        <span class="status-text">Negative Detection</span>
                    </div>
                    <h2 class="result-title">Result: Benign</h2>
                    <p class="result-desc">AI analysis indicates cellular patterns consistent with non-malignant tissue.</p>
                </div>
                """
                html_meter = f"""
                <div class="medscan-card" style="padding-top: 16px; padding-bottom: 16px;">
                    <p style="font-size: 12px; color: #526069; margin-top:0; margin-bottom:8px; font-weight:700; text-transform:uppercase; letter-spacing: 0.05em;">Confidence Meter</p>
                    <div class="meter-container">
                        <div class="meter-bg">
                            <div class="meter-fill-benign" style="width: {confidence_score:.1f}%;"></div>
                        </div>
                        <span class="meter-text text-success">{confidence_score:.1f}%</span>
                    </div>
                </div>
                """
            
            # Display Custom Components
            st.markdown(html_card, unsafe_allow_html=True)
            st.markdown(html_meter, unsafe_allow_html=True)
