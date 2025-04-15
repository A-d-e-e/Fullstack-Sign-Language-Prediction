# Hand Gesture Translation App

An AI-powered full-stack application that translates hand gestures into text and speech in real time. This project leverages computer vision, machine learning, and a modern React-based UI to bridge the communication gap for sign language users.

---

## Table of Contents

- [Features](#features)
- [Project Architecture](#project-architecture)
- [Installation and Setup](#installation-and-setup)
- [Usage](#usage)
- [Future Enhancements](#future-enhancements)
- [Screenshots](#screenshots)
- [Contributing](#contributing)

---

## Features

- **Real-Time Gesture Recognition:** Uses OpenCV and MediaPipe to capture and process live video feed from a webcam.
- **Machine Learning Integration:** A convolutional neural network (CNN) built with TensorFlow/Keras classifies predefined gestures ("Hello", "Yes", "No", "Fullstop", "ClearSentence").
  (Used only a few predefined gestures for prototyping, will use ASL Sign Language Dataset in the future for better results and thousands of sign language gestures)
- **Seamless Backend API:** Flask-based REST API communicates with the React frontend, updating the recognized sentence in real time.
- **Responsive React Frontend:** Displays a live camera feed, recognized gestures, and dynamically updates the translation sentence.
- **Special Gesture Functions:** 
  - **Fullstop:** Appends a period and a space to the sentence.
  - **ClearSentence:** Clears the current sentence to start fresh.
- **Modular Design:** Three-tier architecture ensures separation of concerns and scalability.

---

## Project Architecture

### Backend

- **Language & Libraries:** Python, OpenCV, MediaPipe, TensorFlow/Keras, Flask, NumPy.
- **Components:**
  - **Dataset Collection (`create_dataset.py`):** Captures hand gesture images with landmark detection.
  - **Model Training (`train_model.py`):** Trains a CNN model on the collected dataset and evaluates its accuracy.
  - **Real-Time Recognition & API (`app.py`):** Processes live camera input, recognizes gestures, updates a sentence, and exposes REST endpoints (`/sentence` and `/reset`).

### Frontend

- **Language & Libraries:** JavaScript, React.js.
- **Components:**
  - **Camera Component (`Camera.js`):** Uses the WebRTC API (`navigator.mediaDevices.getUserMedia()`) to access the webcam and display the live feed.
  - **Sentence Display (`SentenceDisplay.js`):** Polls the backend for the latest recognized sentence and updates the UI accordingly.
  - **Main App (`App.js`):** Integrates all components and provides a responsive user interface.
- **Integration:** A proxy defined in `package.json` forwards API requests to the Flask backend, ensuring seamless communication.

---

## Installation and Setup

### Prerequisites

- **Python 3.11.9** and **pip**
- **Node.js** and **npm**

### Backend Setup

1. Navigate to the `backend` folder:
   ```bash
   cd backend
   ```
2. Install Python dependencies:
   ```bash
   pip install opencv-python mediapipe tensorflow flask numpy
   ```
3. Create the required directories:
   - **dataset/** with subfolders: `Hello`, `Yes`, `No`, `Fullstop`, `ClearSentence`
   - **model/** to store the trained model.

### Frontend Setup

1. Navigate to the `frontend` folder:
   ```bash
   cd frontend
   ```
2. Install Node dependencies:
   ```bash
   npm install
   ```
3. The `package.json` is configured with a proxy to `http://localhost:5000`.

---

## Usage

### Running the Backend

1. **Dataset Collection:**  
   Run the dataset collection script to capture hand gesture images:
   ```bash
   python create_dataset.py
   ```
   - Follow on-screen prompts. Press **c** to capture an image and **q** to quit.
   ### Custom Dataset Collection:
   | "How are" | "You" |
   |-----------|-------|
   | ![Custom Dataset Collection: "How are"](https://github.com/user-attachments/assets/5989bbda-5b48-4b32-8f2a-2a45723bd046) | ![Custom Dataset Collection: "You"](https://github.com/user-attachments/assets/00112fe2-3d7f-4553-936e-02596981c6a0) |

   ### Live Feed during data collection:

   | "Capturing 'Hello'" |
   |----------------------|
   | ![Live feed create data](https://github.com/user-attachments/assets/695e9d8e-e863-4153-8891-edd828b54a25) |

     
     
3. **Model Training:**  
   Train the model on your collected dataset:
   ```bash
   python train_model.py
   ```
   - Optionally, test the model live:
   ```bash
   python train_model.py test
   ```

4. **Start the Flask Server:**  
   Run the Flask backend:
   ```bash
   python app.py
   ```
   - This will open a live "Recognition" window displaying the camera feed and gesture recognition.

### Running the Frontend

1. In a separate terminal, start the React app:
   ```bash
   npm start
   ```
2. The React app will launch in your browser (typically at `http://localhost:3000`) and automatically communicate with the backend.

### Integrated Operation

With both the backend and frontend running, the system functions as follows:
- The backend continuously processes camera input, recognizes gestures, and updates a global sentence.
- The React frontend polls the `/sentence` endpoint and displays the updated sentence along with a live camera feed.
- Special gestures ("Fullstop" and "ClearSentence") are handled appropriately to update the sentence.

---

## Future Enhancements

- **Model Improvement:** Expand training on a larger ASL dataset and refine the CNN model for higher accuracy.
- **UI/UX Enhancements:** Improve the visual design of the React app with modern CSS frameworks and responsive design.
- **Deployment:** Containerize the application with Docker and deploy on cloud platforms like Heroku or AWS.
- **Additional Features:** Integrate text-to-speech functionality, user authentication, and detailed logging.

---

## Screenshots

- **Backend Recognition Window:** 
- **React App Homepage:** 
- **Terminal Logs:** 


## Hosting and Deployment
- Here's what I plan for the future:
- Once the model is trained on a large dataset, the project will be deployed over the internet.
- 

## Contributing

Contributions are welcome! Please fork the repository, create a new branch for your feature or bug fix, and submit a pull request. Ensure your code is well-documented and tested.
