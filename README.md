# traffic-sign-recognition
This project is a machine learning–based traffic sign classifier. It uses a trained convolutional neural network to recognize 43 different types of traffic signs.  

You can test the model in two ways:  
- A **Python console script** with a file dialog (using Tkinter)  
- A **Streamlit web application** for a more interactive and modern user experience  

---

## 📁 Project Structure

traffic-sign-recognition/
│
├── traffic_sign_model.h5 # trained model
├── traffic_sign_app.py # Streamlit app
├── test_traffic_sign.py # console-based test script
├── requirements.txt # dependencies
├── uploads/ # (optional) store uploaded images
└── README.md

yaml
Copy
Edit

---

## 🚀 Features

- Predicts 43 classes of German traffic signs  
- Accepts any `.jpg`, `.png`, or `.jpeg` image  
- Uses TensorFlow (Keras) deep learning  
- Simple and professional user interface with Streamlit  
- Option to store uploaded images  

---
## 🖼️ Screenshots

### Streamlit Interface
![Screenshot 2025-06-28 164130](https://github.com/user-attachments/assets/55a6eae9-e89e-46c0-b5c3-998b12d53c8f)




### Prediction Example

![Screenshot 2025-06-28 164357](https://github.com/user-attachments/assets/a4c2fefc-ed20-4ee4-bab0-fcaedbf512e2)

## 🧩 How it works

1. Loads a trained CNN model (`traffic_sign_model.h5`).  
2. Takes an uploaded traffic sign image.  
3. Preprocesses the image (resizes to 32×32, normalizes pixel values).  
4. Predicts the most probable traffic sign class.  
5. Displays the image and its prediction (either in console or Streamlit).

---

## 🖥️ How to run

### 1️⃣ Console-based 

```bash
python test_traffic_sign.py
A file dialog will pop up to choose an image, and the prediction will print in the terminal.

2️⃣ Web-based (Streamlit)
bash
Copy
Edit
streamlit run traffic_sign_app.py
A nice web interface will open in your browser where you can upload an image and see results.

⚙️ Requirements
You can install the dependencies with:

bash
Copy
Edit
pip install -r requirements.txt



