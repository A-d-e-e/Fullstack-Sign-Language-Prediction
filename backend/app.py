# backend/app.py

from flask import Flask, jsonify
import cv2
import mediapipe as mp
import numpy as np
import threading
import tensorflow as tf
import time
import os

app = Flask(__name__)

sentence = ""

MODEL_PATH = os.path.join(os.getcwd(), "..", "model", "hand_gesture_model.h5")
model = tf.keras.models.load_model(MODEL_PATH)

IMG_SIZE = 64
GESTURES = ["Hello", "Yes", "No", "Fullstop", "ClearSentence"]

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

running = True

def gesture_recognition():
    global sentence, running
    cap = cv2.VideoCapture(1)
    with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8) as hands:
        while running:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(cv2.resize(frame, (675, 375)), cv2.COLOR_BGR2RGB)
            rgb.flags.writeable = False
            results = hands.process(rgb)
            rgb.flags.writeable = True
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            
            gesture = None
            if results.multi_hand_landmarks:
                for handLms in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(rgb, handLms, mp_hands.HAND_CONNECTIONS)
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
                            # Display gesture on the frame
                            cv2.putText(rgb, gesture, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
                        except Exception as e:
                            print("Prediction error:", e)
            cv2.imshow("Recognition", rgb)
            key = cv2.waitKey(1)
            if key & 0xFF == ord('q'):
                running = False
                break
            
            # If a gesture is recognized, update the sentence
            if gesture:
                if gesture == "ClearSentence":
                    sentence = ""
                elif gesture == "Fullstop":
                    sentence += ". "
                else:
                    sentence += gesture + " "
                print("Current Sentence:", sentence)
                time.sleep(2)  # wait a second between recognitions to avoid rapid repeats

    cap.release()
    cv2.destroyAllWindows()

# Start the recognition in a background thread.
recognition_thread = threading.Thread(target=gesture_recognition)
recognition_thread.daemon = True
recognition_thread.start()

@app.route("/sentence", methods=["GET"])
def get_sentence():
    return jsonify({"sentence": sentence})

@app.route("/reset", methods=["GET"])
def reset_sentence():
    global sentence
    sentence = ""
    return jsonify({"sentence": sentence, "message": "Sentence cleared"})

if __name__ == "__main__":
    try:
        app.run(host="127.0.0.1", port=5000, debug=False)
    finally:
        running = False
        recognition_thread.join()
