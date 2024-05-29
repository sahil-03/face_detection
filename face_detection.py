import cv2
import os
import datetime
import numpy as np

from PIL import Image
from svd import compute_svd
from svd import compute_alpha
from svd import reconstruct_test_face
from data_loader import construct_X
from data_loader import XInput


OUTPUT_DIR = 'captured_faces'


def capture_face() -> str:
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
            break

        # Display the resulting frame
        cv2.imshow('Webcam - Face Detection', frame)

    # When everything is done, release the capture
    cap.release()
    cv2.destroyAllWindows()

    return output_path


def main():
    # ---- Caputer face ----
    print('Capturing face...')
    # output_path = capture_face()
    output_path = 'captured_faces/face_2024-05-29_10-16-19.png'

    # ---- Load data ----
    input = construct_X()

    # ---- Compute SVD ----
    print('Computing SVD...')
    U, S, Vt = compute_svd(input.X)

    # ---- Compute alpha ----
    with Image.open(output_path) as img: 
        img_array = np.array(img)
        x = img_array.flatten()
    r = U.shape[1]
    alpha = compute_alpha(x, U, r)

    original_mem = input.X.nbytes 
    compressed_mem = U[:, :r].nbytes + alpha.nbytes
    print(f'Compressed memory is ~ {100 * (compressed_mem / original_mem):.2f}% of the original. ')

    # ---- Reconstruct test face ----
    print('Reconstructing test face...')
    reconstruct_test_face(alpha, U, input.avg_face, r)




if __name__ == '__main__':
    main()