import streamlit as st
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

# Load your model
model = load_model('traffic_sign_model.h5')

# Traffic sign class names
class_names = [
    "Speed Limit (20km/h)", "Speed Limit (30km/h)", "Speed Limit (50km/h)",
    "Speed Limit (60km/h)", "Speed Limit (70km/h)", "Speed Limit (80km/h)",
    "End of Speed Limit (80km/h)", "Speed Limit (100km/h)", "Speed Limit (120km/h)",
    "No Overtaking", "No Overtaking for Vehicles Over 3.5 Tons", "Right-of-Way at Intersection",
    "Priority Road", "Yield", "Stop", "No Vehicles", "Vehicles Over 3.5 Tons Prohibited",
    "No Entry", "General Caution", "Dangerous Curve Left", "Dangerous Curve Right",
    "Double Curve", "Bumpy Road", "Slippery Road", "Road Narrows on the Right",
    "Road Work", "Traffic Signals", "Pedestrians", "Children Crossing", "Bicycles Crossing",
    "Beware of Ice/Snow", "Wild Animals Crossing", "End of All Restrictions",
    "Turn Right Ahead", "Turn Left Ahead", "Ahead Only", "Go Straight or Right",
    "Go Straight or Left", "Keep Right", "Keep Left", "Roundabout Mandatory",
    "End of No Overtaking", "End of No Overtaking (Vehicles Over 3.5 Tons)"
]

# --- Page Setup ---
st.set_page_config(page_title="Traffic Sign Recognition", page_icon="🚦", layout="centered")

# --- Custom CSS Styling ---
st.markdown(
    """
    <style>
    html, body, [class*="css"]  {
        font-family: 'Segoe UI', sans-serif;
        background-color: #f0f2f6;
    }
    .title {
        font-size: 48px;
        text-align: center;
        font-weight: 700;
        color: #1A5276;
        margin-bottom: 10px;
    }
    .subtitle {
        text-align: center;
        font-size: 20px;
        color: #5D6D7E;
        margin-bottom: 40px;
    }
    .prediction-box {
        background-color: #D6EAF8;
        padding: 30px 20px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        text-align: center;
    }
    .prediction-box h3 {
        font-size: 24px;
        color: #154360;
    }
    .stButton > button {
        background-color: #2980B9;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
    }
    .stButton > button:hover {
        background-color: #1B4F72;
        color: #fff;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# --- Title ---
st.markdown('<div class="title">🚦 Traffic Sign Recognition</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">Upload an image of a traffic sign to see what it means.</div>', unsafe_allow_html=True)

# --- Upload image ---
uploaded_file = st.file_uploader("Upload a traffic sign image", type=["jpg", "png", "jpeg"])

if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    resized_image = image.resize((32, 32))
    image_array = np.array(resized_image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array)
    predicted_index = np.argmax(prediction)
    confidence = np.max(prediction) * 100
    predicted_label = class_names[predicted_index] if predicted_index < len(class_names) else "Unknown"

    # Layout: 2 columns
    col1, col2 = st.columns([1, 1])

    with col1:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    with col2:
        st.markdown('<div class="prediction-box">', unsafe_allow_html=True)
        st.markdown(f"<h3>🧠 Prediction:</h3><p><b>{predicted_label}</b></p>", unsafe_allow_html=True)
        st.markdown(f"<h3>🔍 Confidence:</h3><p><b>{confidence:.2f}%</b></p>", unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("👈 Upload a traffic sign image to begin.")
