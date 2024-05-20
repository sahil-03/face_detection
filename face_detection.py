import cv2
import os
import datetime

OUTPUT_DIR = 'captured_faces'

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()


while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting...")
        break
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)    
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    if len(faces) > 0:
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
            face_resized = cv2.resize(gray[y:y+h, x:x+w], (168, 192))

            # Save captured image
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            output_path = os.path.join(OUTPUT_DIR, f'face_{timestamp}.png')
            cv2.imwrite(output_path, face_resized)
            print(f'Saved: {output_path}')

        break

    # Display the resulting frame
    cv2.imshow('Webcam - Face Detection', frame)

# When everything is done, release the capture
cap.release()
cv2.destroyAllWindows()
