# backend/train_model.py

import os
import cv2
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
import tensorflow as tf

# Settings
IMG_SIZE = 64
DATASET_PATH = os.path.join(os.getcwd(),"..", "dataset")
GESTURES = ["Hello", "Yes", "No", "Fullstop", "ClearSentence", "How are", "You"]
NUM_CLASSES = len(GESTURES)

def load_data():
    images = []
    labels = []
    for idx, gesture in enumerate(GESTURES):
        gesture_path = os.path.join(DATASET_PATH, gesture)
        for file in os.listdir(gesture_path):
            file_path = os.path.join(gesture_path, file)
            img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            images.append(img)
            labels.append(idx)
    images = np.array(images, dtype="float32") / 255.0
    images = np.expand_dims(images, -1)  
    labels = to_categorical(labels, NUM_CLASSES)
    return images, labels

def build_model():
    model = Sequential([
        Conv2D(32, (3,3), activation="relu", input_shape=(IMG_SIZE, IMG_SIZE, 1)),
        MaxPooling2D((2,2)),
        Conv2D(64, (3,3), activation="relu"),
        MaxPooling2D((2,2)),
        Flatten(),
        Dense(128, activation="relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax")
    ])
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    return model

def train():
    X, y = load_data()
    print("Dataset loaded:", X.shape, y.shape)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model = build_model()
    model.fit(X_train, y_train, epochs=10, validation_data=(X_test, y_test))
    loss, acc = model.evaluate(X_test, y_test)
    print("Test Accuracy:", acc)
    # Save model

    model_save_path = os.path.join(os.getcwd(), "model", "hand_gesture_model.h5")
    print(model_save_path)
    model.save(model_save_path)
    print("Model saved at", model_save_path)

def live_test():
    
    model_path = os.path.join(os.getcwd(), "model", "hand_gesture_model.h5")
    model = tf.keras.models.load_model(model_path)
    cap = cv2.VideoCapture(1)
    mp_hands = __import__("mediapipe").solutions.hands
    mp_drawing = __import__("mediapipe").solutions.drawing_utils

    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8) as hands:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(rgb, handLms, mp_hands.HAND_CONNECTIONS)
                    # Compute bounding box from landmarks
                    h, w, _ = frame.shape
                    lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
                    x_vals = [pt[0] for pt in lm_list]
                    y_vals = [pt[1] for pt in lm_list]
                    x, y = max(0, min(x_vals)-20), max(0, min(y_vals)-20)
                    w_box = min(w, max(x_vals)+20) - x
                    h_box = min(h, max(y_vals)+20) - y
                    if w_box > 0 and h_box > 0:
                        crop_img = frame[y:y+h_box, x:x+w_box]
                        try:
                            gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                            gray = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
                            gray = gray.astype("float32")/255.0
                            gray = np.expand_dims(gray, axis=(0,-1))
                            pred = model.predict(gray)
                            gesture_idx = np.argmax(pred)
                            gesture = GESTURES[gesture_idx]
                            cv2.putText(rgb, gesture, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        except Exception as e:
                            print("Prediction error:", e)
            cv2.imshow("Live Test", cv2.resize(rgb, (900, 500)))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "test":
        live_test()
    else:
        train()

