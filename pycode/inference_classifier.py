import pickle
import cv2
import mediapipe as mp
import numpy as np
from gtts import gTTS
from playsound import playsound
import threading

# Load the trained model
model_dict = pickle.load(open('./pycode/model.p', 'rb'))
model = model_dict['model']

# Initialize video capture using the default camera (index 0)
cap = cv2.VideoCapture(0)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Unable to open camera.")
    exit()

# Set the desired frame width and height (adjust as needed)
frame_width = 320
frame_height = 240
cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.4)

labels_dict = {0: 'Hello', 1: 'BOSTON', 2: 'I', 3: "LOVE",  4: "Hackathon"}

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

# Initialize the frame variable outside the loop
frame = None

# Lock for thread safety
frame_lock = threading.Lock()

# Flag to indicate if the close button is clicked
exit_clicked = False

# Initialize variables for tracking the last detected sign
last_detected_sign = None

def process_frames():
    global frame
    global cap
    global exit_clicked
    global last_detected_sign

    frame_counter = 0
    skip_frames = 2  # Adjust as needed

    while not exit_clicked:
        # Read a frame from the camera
        ret, current_frame = cap.read()

        # Skip frames
        frame_counter += 1
        if frame_counter % skip_frames != 0:
            continue

        # Process hand landmarks using MediaPipe
        current_frame_rgb = cv2.cvtColor(current_frame, cv2.COLOR_BGR2RGB)
        results = hands.process(current_frame_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    current_frame,
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

                # Check if the detected sign is different from the last one
                current_detected_sign = int(predicted_label[0])
                if current_detected_sign != last_detected_sign:
                    # Draw the predicted label on the frame
                    label_text = labels_dict[current_detected_sign]

                    # Calculate the x-coordinate for each label to avoid overlapping
                    text_width = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 2, 3)[0][0]
                    x_coord = int((current_frame.shape[1] - text_width) / 2)  # Center the text
                    cv2.putText(current_frame, label_text, (x_coord, 100), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

                    # Speak the predicted label in Nepali using gTTS
                    tts = gTTS(text=label_text, lang='en')  
                    tts.save("output.mp3")
                    playsound("output.mp3")

                    # Update the last detected sign
                    last_detected_sign = current_detected_sign

        # Update the global frame with the processed frame
        with frame_lock:
            frame = current_frame.copy()

# Start the processing thread
processing_thread = threading.Thread(target=process_frames)
processing_thread.start()

while not exit_clicked:
    # Check if the frame is not None and has valid dimensions
    with frame_lock:
        if frame is not None and frame.shape[0] > 0 and frame.shape[1] > 0:
            # Display the frame
            cv2.imshow('frame', frame)

    # Add a delay to control the frame rate (adjust as needed)
    key = cv2.waitKeyEx(30)

    # Check if the user pressed 'q' or the close window button
    if key == 27:  # 27 is the ASCII code for the 'Esc' key
        exit_clicked = True

# Release the camera and close the OpenCV window
cap.release()
cv2.destroyAllWindows()
