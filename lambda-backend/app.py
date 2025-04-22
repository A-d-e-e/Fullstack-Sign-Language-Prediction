import json
import base64
import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import time
import os
import logging

# Set up logging for debugging
logging.basicConfig(level=logging.INFO)

# Global sentence variable (for demo only – note that Lambda is stateless, so consecutive invocations
# may not share state reliably in production)
sentence = ""

# Load the trained model from the "model" folder (make sure to include this in your zip)
MODEL_PATH = os.path.join(os.getcwd(), "model", "hand_gesture_model.h5")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    logging.info("Model loaded successfully")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Could not load model") from e

# Constants
IMG_SIZE = 64
GESTURES = ["Hello", "Yes", "No", "Fullstop", "ClearSentence"]

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8)

def process_image(image_data):
    """
    Process a base64 encoded image to extract a hand region,
    predict a gesture using your model, and return that gesture.
    """
    try:
        # Expecting the image data in the format: "data:image/jpeg;base64,<encoded-data>"
        header, encoded = image_data.split(",", 1)
        image_bytes = base64.b64decode(encoded)
        nparr = np.frombuffer(image_bytes, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            logging.error("Frame is None after decoding")
            return None

        # Flip the frame to mimic a mirror effect
        frame = cv2.flip(frame, 1)

        # Convert to RGB for MediaPipe processing
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb_frame)
        if not results.multi_hand_landmarks:
            logging.info("No hand landmarks detected in the current frame")
            return None

        # Process only the first detected hand
        hand_landmarks = results.multi_hand_landmarks[0]
        h, w, _ = frame.shape
        lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in hand_landmarks.landmark]
        x_vals = [pt[0] for pt in lm_list]
        y_vals = [pt[1] for pt in lm_list]
        x = max(0, min(x_vals) - 20)
        y = max(0, min(y_vals) - 20)
        w_box = min(w, max(x_vals) + 20) - x
        h_box = min(h, max(y_vals) + 20) - y

        if w_box <= 0 or h_box <= 0:
            logging.error("Invalid bounding box dimensions")
            return None

        crop_img = frame[y:y+h_box, x:x+w_box]
        gray = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
        resized = cv2.resize(gray, (IMG_SIZE, IMG_SIZE))
        normalized = resized.astype("float32") / 255.0
        input_data = np.expand_dims(normalized, axis=(0, -1))
        pred = model.predict(input_data)
        gesture_idx = int(np.argmax(pred))
        gesture = GESTURES[gesture_idx]
        logging.info(f"Predicted gesture: {gesture}")
        return gesture

    except Exception as e:
        logging.error(f"Error in process_image: {e}")
        return None

def lambda_handler(event, context):
    """
    AWS Lambda handler function.
    
    Expects the incoming event to have a JSON body with a key "image" that is a base64 encoded image.
    Updates the global sentence variable based on the recognized gesture.
    """
    global sentence
    try:
        body = json.loads(event.get("body", "{}"))
        image_data = body.get("image")
        if not image_data:
            raise ValueError("No image provided in the request")
        
        gesture = process_image(image_data)
        
        # Update the global sentence variable:
        if gesture:
            if gesture == "ClearSentence":
                sentence = ""
            elif gesture == "Fullstop":
                sentence += ". "
            else:
                sentence += gesture + " "
        
        response_body = {"gesture": gesture, "sentence": sentence}
        return {
            "statusCode": 200,
            "headers": {
                "Content-Type": "application/json",
                "Access-Control-Allow-Origin": "*"  # Required for CORS
            },
            "body": json.dumps(response_body)
        }
    except Exception as e:
        logging.error(f"Error in lambda_handler: {e}")
        return {
            "statusCode": 500,
            "headers": {"Access-Control-Allow-Origin": "*"},
            "body": json.dumps({"error": str(e)})
        }
