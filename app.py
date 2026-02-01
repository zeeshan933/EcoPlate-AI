import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json
import os
import pandas as pd
import hashlib
from datetime import datetime
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ==========================================
# 1. PAGE CONFIGURATION & SESSION STATE
# ==========================================
st.set_page_config(
    page_title="EcoPlate: AI Waste Tracker",
    page_icon="🍽️",
    layout="wide"
)

# Initialize Session State for History and ID Tracking
if 'history' not in st.session_state:
    st.session_state['history'] = []
if 'last_logged_id' not in st.session_state:
    st.session_state['last_logged_id'] = None

# --- CSS to hide Streamlit style elements ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}

            [data-testid="stSidebar"] {
                min-width: 200px;
                max-width: 250px;
            }
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR SETTINGS
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Upload Image", "Live Analysis"])
st.sidebar.markdown("---")

# --- A. Calibration ---
with st.sidebar.expander("⚙️ Advanced Calibration", expanded=False):
    st.write("Adjust these if the results are inaccurate.")
    show_debug = st.checkbox("Show AI Vision (Debug)", value=False)
    thresh_offset = st.slider("Shadow Sensitivity", -50, 50, -15)

# --- B. Business Settings ---
st.sidebar.markdown("### 💼 Business Settings")
plate_cost = st.sidebar.number_input("Cost per Plate (Rs)", min_value=0.0, value=15.0, step=0.5)

# --- C. Session Statistics ---
st.sidebar.markdown("---")
st.sidebar.markdown("### 📊 Session Stats")

if st.session_state['history']:
    # Convert list of dicts to DataFrame
    df = pd.DataFrame(st.session_state['history'])
    
    # Display Key Metrics in Sidebar
    total_waste = df['Waste %'].mean()
    total_money = df['Money Lost (Rs)'].sum()
    st.sidebar.metric("Avg Waste", f"{total_waste:.1f}%")
    st.sidebar.metric("Total Loss", f"Rs{total_money:.2f}")
    
    # Download Button
    csv = df.to_csv(index=False).encode('utf-8')
    st.sidebar.download_button(
        "📥 Download Report (CSV)",
        csv,
        "food_waste_report.csv",
        "text/csv",
        key='download-csv'
    )
else:
    st.sidebar.info("No plates analyzed yet.")

# ==========================================
# 3. LOAD MODEL & DATA
# ==========================================
@st.cache_resource
def load_resources():
    # PATH CONFIGURATION
    model_path = 'Models/EfficientNetB3_food_waste_model.h5' 
    
    # Fallback path logic
    if not os.path.exists(model_path):
        model_path = '/home/muhammad-zeeshan/Desktop/AI Term Project/Models/EfficientNetB3_food_waste_model.h5'

    try:
        model = tf.keras.models.load_model(model_path)
    except:
        return None, None
    
    default_labels = [
        "Bread", "Dairy product", "Dessert", "Egg", "Fried food", 
        "Meat", "Noodles-Pasta", "Rice", "Seafood", "Soup", "Vegetable-Fruit"
    ]
    
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        labels = list(class_indices.keys())
    except:
        labels = default_labels
        
    return model, labels

model, LABELS = load_resources()

# --- CONSTANTS ---
CO2_FACTORS = {
    "Meat": 20.0, "Seafood": 12.0, "Dairy product": 8.0, "Egg": 4.8,
    "Fried food": 3.5, "Rice": 2.7, "Dessert": 2.5, "Noodles-Pasta": 1.5,
    "Bread": 1.2, "Soup": 1.0, "Vegetable-Fruit": 0.5, "None": 0.0
}

PORTION_STANDARDS = {
    "Rice": 0.65, "Noodles-Pasta": 0.65, "Vegetable-Fruit": 0.60,
    "Meat": 0.45, "Fried food": 0.50, "Seafood": 0.45,
    "Bread": 0.40, "Dessert": 0.30, "Dairy product": 0.40,
    "Soup": 0.70, "Egg": 0.30, "None": 1.0
}

# ==========================================
# 4. CORE LOGIC (Generates ID)
# ==========================================
def process_image(image_input, sensitivity_offset=0):
    if isinstance(image_input, Image.Image):
        img_array = np.array(image_input)
    else:
        img_array = image_input

    # 1. Generate Unique ID for this image (Hash of pixels)
    # This helps us distinguish between a "New Image" and a "UI Refresh"
    img_bytes = img_array.tobytes()
    image_id = hashlib.md5(img_bytes).hexdigest()

    # 2. Pre-processing
    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_array, (224, 224))
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)

    # 3. Empty Check
    if np.std(gray) < 15:
        return img_array, "None", 0.0, 0, 1, np.zeros_like(gray), image_id

    # 4. AI Prediction
    img_batch = np.expand_dims(img_resized, axis=0)
    img_batch = preprocess_input(img_batch)
    predictions = model.predict(img_batch)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    food_name = "None" if confidence < 45 else LABELS[class_idx]

    # 5. Segmentation
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    otsu_thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    _, thresh = cv2.threshold(blurred, otsu_thresh_val + sensitivity_offset, 255, cv2.THRESH_BINARY_INV)

    h, w = thresh.shape
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (w//2, h//2), int(h * 0.45), 255, -1)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)
    
    kernel = np.ones((4,4), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    food_pixels = cv2.countNonZero(thresh)
    mask_area = np.pi * (int(h * 0.45) ** 2)
    
    return img_array, food_name, confidence, food_pixels, mask_area, thresh, image_id

# ==========================================
# 5. UI COMPONENTS (Auto & Manual Overwrite Logic)
# ==========================================
def display_results(food_name, conf, food_pixels, mask_area, debug_mask, image_id):
    st.subheader("📝 Analysis Results")
    
    # 1. Correction Input
    try:
        idx = LABELS.index(food_name)
    except ValueError:
        idx = 0
        
    verified_food = st.selectbox("Detected Food Item:", LABELS, index=idx)
    
    # --- CONFIDENCE DISPLAY ---
    st.info(f"🤖 AI Confidence Score: **{conf:.1f}%**")

    # 2. Calculate Metrics
    standard_ratio = PORTION_STANDARDS.get(verified_food, 0.60) 
    
    if mask_area > 0:
        waste_pct = (food_pixels / (mask_area * standard_ratio)) * 100
        waste_pct = min(max(waste_pct, 0), 100)
        rec = "Portion is Optimal"
        if waste_pct > 40: rec = "Consider Smaller Portion"
        elif waste_pct < 5: rec = "Plate Cleaned"
    else:
        waste_pct = 0
        rec = "Empty"

    money_lost = (waste_pct / 100) * plate_cost
    wasted_kg = 0.5 * (waste_pct / 100)
    co2_emit = wasted_kg * CO2_FACTORS.get(verified_food, 2.0)

    # 3. Create Data Entry
    current_entry = {
        "Time": datetime.now().strftime("%H:%M:%S"),
        "Food Item": verified_food,
        "Waste %": round(waste_pct, 1),
        "Money Lost (Rs)": round(money_lost, 2),
        "CO2 (kg)": round(co2_emit, 2),
        "Confidence": round(conf, 1),
        "ID": image_id
    }

    # --- AUTOMATIC LOGGING (Overwrite if Exists) ---
    # Check if ID exists in history
    auto_index = next((i for i, d in enumerate(st.session_state['history']) if d["ID"] == image_id), None)

    if auto_index is not None:
        # FOUND: Overwrite existing entry
        st.session_state['history'][auto_index] = current_entry
    else:
        # NOT FOUND: Add new entry
        st.session_state['history'].append(current_entry)
        st.session_state['last_logged_id'] = image_id
        st.toast("✅ Auto-saved new plate!")

    # --- MANUAL BUTTON (Overwrite if Exists) ---
    if st.button("💾 Manual Save / Update"):
        # Check if ID exists (It likely does due to Auto-Log, but we check to be safe)
        manual_index = next((i for i, d in enumerate(st.session_state['history']) if d["ID"] == image_id), None)
        
        if manual_index is not None:
            # FOUND: Delete old (overwrite)
            st.session_state['history'][manual_index] = current_entry
            st.success("✅ Entry updated successfully!")
        else:
            # NOT FOUND: Add new
            st.session_state['history'].append(current_entry)
            st.success("✅ New entry saved!")

    # 4. Display Visuals
    m1, m2, m3 = st.columns(3)
    m1.metric("Waste Amount", f"{waste_pct:.1f}%", delta="-High" if waste_pct > 30 else "Low")
    m2.metric("Money Lost", f"Rs{money_lost:.2f}", delta_color="inverse")
    m3.metric("CO2 Footprint", f"{co2_emit:.2f} kg")

    st.progress(int(waste_pct))
    if waste_pct > 30:
        st.error(f"⚠️ Recommendation: {rec}")
    else:
        st.success(f"✅ Recommendation: {rec}")

    if show_debug:
        st.warning("🤖 Computer Vision Mask")
        st.image(debug_mask, caption="White = Detected Waste", width=200)


# ==========================================
# 6. APP EXECUTION
# ==========================================
st.title("🍽️ EcoPlate: Smart Waste Tracker")
st.markdown("---")

if model is None:
    st.error("❌ Model not found! Please check path configurations.")
    st.stop()

# --- MODE 1: UPLOAD ---
if app_mode == "Upload Image":
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload")
        uploaded_file = st.file_uploader("Choose a plate image...", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        if uploaded_file:
            # Process returns ID now
            _, food_name, conf, food_pixels, mask_area, debug_mask, img_id = process_image(image, thresh_offset)
            # Display automatically handles the saving
            display_results(food_name, conf, food_pixels, mask_area, debug_mask, img_id)

# --- MODE 2: LIVE ANALYSIS ---
elif app_mode == "Live Analysis":
    st.subheader("📷 Live Camera")
    img_file_buffer = st.camera_input("Capture Plate")

    if img_file_buffer is not None:
        image = Image.open(img_file_buffer)
        
        # Analyze
        _, food_name, conf, food_pixels, mask_area, debug_mask, img_id = process_image(image, thresh_offset)
        
        # Overlay Logic
        temp_waste = 0
        if mask_area > 0:
            temp_waste = (food_pixels / (mask_area * 0.65)) * 100
        
        annotated_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = annotated_img.shape
        cv2.putText(annotated_img, f"{food_name}: {temp_waste:.1f}%", (20, h-40), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        
        st.image(cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB), caption="Analyzed", use_container_width=True)
        
        st.markdown("---")
        # Automatic Logging occurs here
        display_results(food_name, conf, food_pixels, mask_area, debug_mask, img_id)