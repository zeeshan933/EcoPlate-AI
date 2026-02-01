import streamlit as st
import tensorflow as tf
import cv2
import numpy as np
from PIL import Image
import json
import os
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="EcoPlate: AI Waste Tracker",
    page_icon="🍽️",
    layout="wide"
)

# --- CSS to hide Streamlit style elements (Corrected) ---
hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            /* header {visibility: hidden;} <--- COMMENTED OUT so sidebar arrow stays visible */
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# ==========================================
# 2. SIDEBAR SETTINGS (Updated with Hide/Unhide)
# ==========================================
st.sidebar.title("Navigation")
app_mode = st.sidebar.radio("Go to:", ["Upload Image", "Live Analysis"])

st.sidebar.markdown("---")

# We use st.expander to create the "Hide/Unhide" effect
with st.sidebar.expander("⚙️ Advanced Calibration", expanded=False):
    st.write("Adjust these if the results are inaccurate.")
    
    # These controls are now hidden inside the expander
    show_debug = st.checkbox("Show AI Vision (Debug)", value=False, help="See the black & white mask used for counting.")
    
    thresh_offset = st.slider(
        "Shadow Sensitivity", 
        min_value=-50, 
        max_value=50, 
        value=-15, 
        help="Slide RIGHT if it misses food. Slide LEFT if it thinks shadows are food."
    )
# ==========================================
# 3. LOAD MODEL & RESOURCES (Cached)
# ==========================================
@st.cache_resource
def load_resources():
    # 1. Load Model
    # Note: Ensure the path is correct for your specific machine
    model_path = 'EfficientNetB3_food_waste_model.h5'
    
    try:
        model = tf.keras.models.load_model(model_path)
    except:
        st.error(f"❌ Error: Could not load model from '{model_path}'. Please check the file path.")
        return None, None
    
    # 2. Load Labels
    # We try to load from file first, but if missing, we use this Hardcoded List
    # This matches the 11 classes from your Confusion Matrix
    default_labels = [
        "Bread", 
        "Dairy product", 
        "Dessert", 
        "Egg", 
        "Fried food", 
        "Meat", 
        "Noodles-Pasta", 
        "Rice", 
        "Seafood", 
        "Soup", 
        "Vegetable-Fruit"
    ]
    
    try:
        with open('class_indices.json', 'r') as f:
            class_indices = json.load(f)
        # Sort keys to ensure order matches the model's training order
        labels = list(class_indices.keys())
    except:
        # If file is missing, use the hardcoded list above
        labels = default_labels
        
    return model, labels

model, LABELS = load_resources()

# ==========================================
# 4. CORE LOGIC (Updated with 'None')
# ==========================================
def process_image(image_input, sensitivity_offset=0):
    # ... (Standard loading code) ...
    if isinstance(image_input, Image.Image):
        img_array = np.array(image_input)
    else:
        img_array = image_input

    img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_array, (224, 224))

    # --- A. EMPTY PLATE CHECK (Force "None") ---
    # If the image is flat/empty (low variance), don't even ask the model.
    gray = cv2.cvtColor(img_resized, cv2.COLOR_RGB2GRAY)
    std_dev = np.std(gray)
    
    if std_dev < 15:  # "Clean Plate" Threshold
        # Return "None" immediately
        return img_array, "None", 0.0, 0.0, "Plate is empty", np.zeros_like(gray)

    # --- B. AI PREDICTION ---
    from tensorflow.keras.applications.mobilenet_v3 import preprocess_input
    img_batch = np.expand_dims(img_resized, axis=0)
    img_batch = preprocess_input(img_batch)

    predictions = model.predict(img_batch)
    class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    # --- C. CONFIDENCE CHECK (Force "None") ---
    # If the AI is less than 45% sure, it's probably looking at a table or wall.
    if confidence < 45:
        food_name = "None"
        waste_pct = 0.0
        rec_text = "No food detected"
        # We can stop here or let it calculate waste purely visually
        # Let's return early to be safe
        return img_array, "None", confidence, 0.0, "No food detected", np.zeros_like(gray)
    else:
        food_name = LABELS[class_idx]

    # --- D. WASTE CALCULATION (Visual) ---
    blurred = cv2.GaussianBlur(gray, (7, 7), 0)
    
    # Thresholding with your slider
    otsu_thresh_val, _ = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    final_thresh_val = otsu_thresh_val + sensitivity_offset
    _, thresh = cv2.threshold(blurred, final_thresh_val, 255, cv2.THRESH_BINARY_INV)

    # Circular Mask
    h, w = thresh.shape
    mask = np.zeros_like(thresh)
    cv2.circle(mask, (w//2, h//2), int(h * 0.45), 255, -1)
    thresh = cv2.bitwise_and(thresh, thresh, mask=mask)

    # Morphology
    kernel = np.ones((4,4), np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    # Calculate
    food_pixels = cv2.countNonZero(thresh)
    mask_area = np.pi * (int(h * 0.45) ** 2)
    standard_portion = mask_area * 0.65 
    
    waste_pct = (food_pixels / standard_portion) * 100
    waste_pct = min(waste_pct, 100)
    if waste_pct < 2: waste_pct = 0

    # Recommendation
    rec_text = "Portion is Optimal"
    if waste_pct > 40:
        rec_text = "Consider Smaller Portion"
    elif waste_pct < 5:
        rec_text = "Plate Cleaned"
    
    return img_array, food_name, confidence, waste_pct, rec_text, thresh
# ==========================================
# 5. APP INTERFACE
# ==========================================

st.title("🍽️ AI Food Waste Tracker")
st.markdown("---")

# CHECK MODEL
if model is None:
    st.error("❌ Model not found! Please upload 'food_waste_model.h5' and 'class_indices.json'.")
    st.stop()

# --- PAGE 1: UPLOAD IMAGE ---
if app_mode == "Upload Image":
    
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Upload")
        uploaded_file = st.file_uploader("Choose a food image...", type=["jpg", "jpeg", "png"])
        
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            st.image(image, caption="Uploaded Image", use_container_width=True)

    with col2:
        st.subheader("2. Analysis Results")
        
        if uploaded_file is not None:
            with st.spinner('Analyzing food waste...'):
                # Process with Slider Value
                _, food_name, conf, waste_pct, rec, debug_mask = process_image(image, sensitivity_offset=thresh_offset)
                
                # DISPLAY METRICS
                st.success(f"**Detected Item:** {food_name}")
                st.info(f"**AI Confidence:** {conf:.1f}%")
                
                # Waste Gauge
                st.metric(label="Estimated Waste", value=f"{waste_pct:.1f}%", delta=f"{'-High' if waste_pct>30 else 'Low'}")
                st.progress(int(waste_pct))
                
                # Recommendation Box
                if waste_pct > 30:
                    st.error(f"⚠️ Recommendation: {rec}")
                else:
                    st.success(f"✅ Recommendation: {rec}")

                # DEBUG VIEW (Only if checkbox is checked)
                if show_debug:
                    st.warning("🤖 Computer Vision Mask:")
                    st.image(debug_mask, caption="White Area = Counted as Waste", width=200)

# --- PAGE 2: LIVE ANALYSIS ---
elif app_mode == "Live Analysis":
    st.subheader("📷 Live Camera Analysis")
    st.write("Take a snapshot of the plate to analyze it instantly.")

    # Streamlit Camera Input
    img_file_buffer = st.camera_input("Capture Plate")

    if img_file_buffer is not None:
        # Load the image
        image = Image.open(img_file_buffer)
        
        # Process with Slider Value
        original_img, food_name, conf, waste_pct, rec, debug_mask = process_image(image, sensitivity_offset=thresh_offset)
        
        # --- DRAWING "IN FRONT" OF PREVIEW ---
        # Convert to BGR for OpenCV drawing
        annotated_img = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        h, w, _ = annotated_img.shape
        
        # Draw Overlay Box at Bottom
        overlay = annotated_img.copy()
        cv2.rectangle(overlay, (0, h-100), (w, h), (0, 0, 0), -1)
        annotated_img = cv2.addWeighted(overlay, 0.6, annotated_img, 0.4, 0)
        
        # Draw Text
        text_color = (0, 255, 0) if waste_pct < 30 else (0, 0, 255)
        cv2.putText(annotated_img, f"{food_name}: {waste_pct:.1f}% Waste", (20, h-60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        cv2.putText(annotated_img, f"Action: {rec}", (20, h-20), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, text_color, 2)
        
        # If Debug Mode is ON, overlay the mask in the corner
        if show_debug:
            mask_small = cv2.resize(debug_mask, (100, 100))
            mask_bgr = cv2.cvtColor(mask_small, cv2.COLOR_GRAY2BGR)
            annotated_img[10:110, w-110:w-10] = mask_bgr # Top right corner

        # Convert back to RGB for Streamlit display
        annotated_img = cv2.cvtColor(annotated_img, cv2.COLOR_BGR2RGB)

        # SHOW THE OUTPUT "IN FRONT" (Replaces preview)
        st.image(annotated_img, caption="Analyzed Capture", use_container_width=True)