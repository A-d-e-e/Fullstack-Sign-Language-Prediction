# backend/create_dataset.py

import cv2
import mediapipe as mp
import os
import time

# List of gestures to capture – adjust as needed.
GESTURES = ["Hello", "Yes", "No", "Fullstop", "ClearSentence", "How are", "You"]

#os.getcwd()

# Path to dataset folder 
DATASET_PATH = os.path.join(os.getcwd(), "dataset")
#print(DATASET_PATH)

# Create folders if they don't exist
for gesture in GESTURES:
    gesture_path = os.path.join(DATASET_PATH, gesture)
    print(gesture_path)
    if not os.path.exists(gesture_path):
        os.makedirs(gesture_path)

# Initialize MediaPipe Hands.
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(1)
print("Enter the gesture label for this capture (choose from {}):".format(GESTURES))
gesture_label = input("Gesture: ").strip()
if gesture_label not in GESTURES:
    print("Invalid gesture label.")
    cap.release()
    exit()

# Counter for image filenames.
img_count = len(os.listdir(os.path.join(DATASET_PATH, gesture_label)))

with mp_hands.Hands(max_num_hands=2, min_detection_confidence=0.8) as hands:
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)  # Mirror effect
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        image.flags.writeable = False
        results = hands.process(image)
        image.flags.writeable = True
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        if results.multi_hand_landmarks:
            for handLms in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, handLms, mp_hands.HAND_CONNECTIONS)
                
                h, w, _ = image.shape
                lm_list = [(int(lm.x * w), int(lm.y * h)) for lm in handLms.landmark]
                x_vals = [pt[0] for pt in lm_list]
                y_vals = [pt[1] for pt in lm_list]
                x, y, w_box, h_box = min(x_vals), min(y_vals), max(x_vals)-min(x_vals), max(y_vals)-min(y_vals)
                cv2.rectangle(image, (x-10, y-10), (x+w_box+10, y+h_box+10), (0, 255, 0), 2)
                
                # Crop the hand region with a little offset.
                crop_img = frame[max(0, y-20):y+h_box+20, max(0, x-20):x+w_box+20]
                
                # Show the cropped image in a separate window.
                cv2.imshow("Cropped Hand", crop_img)
                
        cv2.imshow("Dataset Collection - {}".format(gesture_label), cv2.resize(image, (900, 500)))
        
        key = cv2.waitKey(1)
        if key & 0xFF == ord('c'):
            # Save the cropped image if available
            if results.multi_hand_landmarks:
                save_path = os.path.join(DATASET_PATH, gesture_label, f"{img_count}.jpg")
                cv2.imwrite(save_path, crop_img)
                print(f"Saved {save_path}")
                img_count += 1
                time.sleep(0.3)
        elif key & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
