import os
import cv2

DATA_DIR = './pycode/data'
if not os.path.exists(DATA_DIR):
    os.makedirs(DATA_DIR)

number_of_classes = 5
dataset_size = 120

# Use the correct camera index (0, 1, 2, etc.)
cap = cv2.VideoCapture(0)

for j in range(number_of_classes):
    class_dir = os.path.join(DATA_DIR, str(j))
    if not os.path.exists(class_dir):
        os.makedirs(class_dir)

    print(f'Collecting data for class {j}')

    # Display a message on the screen until 'q' is pressed
    while True:
        ret, frame = cap.read()
        cv2.putText(frame, 'Ready? Press "Q" to start recording.', (50, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.imshow('frame', frame)
        if cv2.waitKey(25) == ord('q'):
            break

    # Start recording frames
    counter = 0
    while counter < dataset_size:
        ret, frame = cap.read()
        cv2.imshow('frame', frame)
        cv2.waitKey(25)

        # Save the frame to the class directory
        img_filename = f'{counter}.jpg'
        img_path = os.path.join(class_dir, img_filename)
        cv2.imwrite(img_path, frame)

        counter += 1

print('Data collection completed.')
cap.release()
cv2.destroyAllWindows()
