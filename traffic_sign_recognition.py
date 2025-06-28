import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.utils import to_categorical

# ✅ Set your dataset path
DATA_DIR = r'C:\Users\user\Downloads\archive\archive\Train'  
IMG_SIZE = 32 

# 🔁 Load and preprocess data
def load_data(data_dir):
    images = []
    labels = []

    # Iterate through each subdirectory (class folder) in the main data directory
    for label in os.listdir(data_dir):
        class_dir = os.path.join(data_dir, label)
        
        # Only process directories (class folders)
        if not os.path.isdir(class_dir):
            continue
        
        # Iterate over each image in the class folder
        for file in os.listdir(class_dir):
            img_path = os.path.join(class_dir, file)
            try:
                # Read image and resize to fixed size (IMG_SIZE x IMG_SIZE)
                image = cv2.imread(img_path)
                image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
                images.append(image)
                labels.append(int(label))  # Use the folder name (label) as the class label
            except Exception as e:
                print(f"Error loading image {img_path}: {e}")
                continue

    return np.array(images), np.array(labels)

# Load dataset
print("[INFO] Loading images...")
X, y = load_data(DATA_DIR)
print(f"Loaded {len(X)} images.")

# 📊 Normalize and one-hot encode labels
X = X / 255.0  # Normalize pixel values to [0, 1]
y = to_categorical(y)  # One-hot encode labels

# 🧪 Train-test split
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 🧱 Build CNN model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_SIZE, IMG_SIZE, 3)),
    MaxPooling2D(pool_size=(2, 2)),
    
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(pool_size=(2, 2)),
    
    Flatten(),
    Dense(128, activation='relu'),
    Dense(y.shape[1], activation='softmax')  # Output layer: number of classes
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 🚀 Train the model
print("[INFO] Training the model...")
history = model.fit(X_train, y_train, epochs=10, validation_data=(X_val, y_val))

# 📈 Plot accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()

# 🧪 Evaluate the model
print("[INFO] Evaluating the model...")
y_pred = model.predict(X_val)
y_pred_classes = np.argmax(y_pred, axis=1)
y_true = np.argmax(y_val, axis=1)

print("[INFO] Classification Report:")
print(classification_report(y_true, y_pred_classes))

# 💾 Save the trained model
model.save("traffic_sign_model.h5")
print("[INFO] Model saved as 'traffic_sign_model.h5'")

