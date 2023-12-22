from absl import logging
logging.get_absl_handler().use_absl_log_file()

import os
import pickle
import mediapipe as mp
import cv2

mp_hands = mp.solutions.hands

hands = mp_hands.Hands(static_image_mode=True, min_detection_confidence=0.5)

DATA_DIR = './data'

# Combine data and labels into a single dictionary
data_dict = {'data': [], 'labels': []}

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

# Iterate over directories in DATA_DIR
for dir_ in os.listdir(DATA_DIR):
    dir_path = os.path.join(DATA_DIR, dir_)

    # Check if dir_path is a directory
    if os.path.isdir(dir_path):
        # Use list comprehension to get image paths
        image_paths = [os.path.join(dir_path, img_path) for img_path in os.listdir(dir_path)]

        for img_path in image_paths:
            data_aux = []
            img = cv2.imread(img_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            results = hands.process(img_rgb)
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    data_aux = process_hand_landmarks(hand_landmarks)
                    data_dict['data'].append(data_aux)
                    data_dict['labels'].append(dir_)

# Save data and labels to a pickle file
with open('data.pickle', 'wb') as f:
    pickle.dump(data_dict, f)
