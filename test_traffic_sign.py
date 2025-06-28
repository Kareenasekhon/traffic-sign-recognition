import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image
from tkinter import Tk, filedialog

# Load your trained modelwhere python

model = load_model('traffic_sign_model.h5')

# Define class names
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

# Function to upload and preprocess image
def upload_image_via_dialog():
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title="Select Traffic Sign Image", filetypes=[("Image files", "*.jpg *.png *.jpeg")])
    
    if not file_path:
        print("No file selected.")
        exit()

    print(f"[INFO] Selected file: {file_path}")
    image = Image.open(file_path).convert("RGB")
    image = image.resize((32, 32))
    image_array = np.array(image) / 255.0
    image_array = np.expand_dims(image_array, axis=0)
    return image_array, file_path

# Predict function
def predict_image(image_array):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Main function
def main():
    print("[INFO] Please select an image for prediction...")
    image_array, file_path = upload_image_via_dialog()

    predicted_class = predict_image(image_array)

    if predicted_class < len(class_names):
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = f"Unknown class ({predicted_class})"

    print(f"[RESULT] Prediction for '{file_path}': {predicted_label}")

# Run main
if __name__ == "__main__":
    main()
