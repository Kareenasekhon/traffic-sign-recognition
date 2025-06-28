# Import necessary libraries
import numpy as np
from tensorflow.keras.models import load_model
from IPython.display import display
import ipywidgets as widgets
import io
from PIL import Image

# Load your trained model
model = load_model('traffic_sign_model.h5')

# Define the 43 traffic sign class names
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

# File upload widget
upload_widget = widgets.FileUpload(accept='image/*', multiple=False)

# Function to upload and preprocess image
def upload_image_jupyter():
    display(upload_widget)

    # Wait for file to be uploaded
    while not upload_widget.value:
        pass

    for filename, fileinfo in upload_widget.value.items():
        print(f"[INFO] Uploaded: {filename}")
        image_data = fileinfo['content']
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        image = image.resize((32, 32))  # Resize to match model input
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)
        return image_array, filename

# Function to make a prediction
def predict_image(image_array):
    prediction = model.predict(image_array)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return predicted_class

# Main function to run prediction
def main():
    print("[INFO] Please upload an image for prediction:")
    image_array, filename = upload_image_jupyter()

    predicted_class = predict_image(image_array)

    if predicted_class < len(class_names):
        predicted_label = class_names[predicted_class]
    else:
        predicted_label = f"Unknown class ({predicted_class})"

    print(f"[RESULT] Prediction for {filename}: {predicted_label}")

# Run the main function
main()
