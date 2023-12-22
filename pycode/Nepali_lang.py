import pickle
import cv2
import mediapipe as mp
from gtts import gTTS
from playsound import playsound
import os
import numpy as np  # Add this import statement

# Load the trained model
model_dict = pickle.load(open('./model.p', 'rb'))
model = model_dict['model']

# Initialize video capture using the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Set the desired frame width and height (adjust as needed)
frame_width = 640
frame_height = 480
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.3)

labels_dict = {0: 'नमस्ते ', 1: 'धन्यवाद', 2: 'सोरी'}

def process_hand_landmarks(hand_landmarks):
    data_aux = []
    x_ = []
    y_ = []

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y

        x_.append(x)
        y_.append(y)

    for i in range(len(hand_landmarks.landmark)):
        x = hand_landmarks.landmark[i].x
        y = hand_landmarks.landmark[i].y
        data_aux.append(x - min(x_))
        data_aux.append(y - min(y_))

    return data_aux

while True:
    # Read a frame from the camera
    ret, frame = cap.read()

    # Check if the frame was read successfully
    if not ret:
        print("Error: Unable to capture frame.")
        continue  # Skip to the next iteration

    # Resize the frame to a smaller size for processing
    frame = cv2.resize(frame, (frame_width, frame_height))

    # Get the height (H), width (W), and channels (C) of the frame
    H, W, _ = frame.shape

    # Convert the frame to RGB format
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process hand landmarks using MediaPipe
    results = hands.process(frame_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style()
            )

            # Process the hand landmarks to get the feature vector
            feature_vector = process_hand_landmarks(hand_landmarks)

            # Reshape the feature vector to match the input shape expected by the model
            feature_vector = np.array(feature_vector).reshape(1, -1)

            # Get the predicted label
            predicted_label = model.predict(feature_vector)

            # Draw the predicted label on the frame
            label_text = labels_dict[int(predicted_label[0])]
            cv2.putText(frame, label_text, (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            # Speak the predicted label in Nepali using gTTS
            tts = gTTS(text=label_text, lang='ne')  # 'ne' is the language code for Nepali
            tts.save("output.mp3")

            # Play the audio using the playsound library
            playsound("output.mp3")

    # Display the frame
    cv2.imshow('frame', frame)

    # Add a delay to control the frame rate (adjust as needed)
    if cv2.waitKey(30) & 0xFF == ord('q'):
        break

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
