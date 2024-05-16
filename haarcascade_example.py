from __future__ import print_function
import cv2 as cv
import argparse

def detectAndDisplay(frame):
    frame_gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    frame_gray = cv.equalizeHist(frame_gray)

    #-- Detect faces
    faces = face_cascade.detectMultiScale(frame_gray)
    for (x, y, w, h) in faces:
        # Draw rectangle around the face
        frame = cv.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 255), 4)
        faceROI = frame_gray[y:y + h, x:x + w]

        #-- In each face, detect eyes
        eyes = eyes_cascade.detectMultiScale(faceROI)
        for (x2, y2, w2, h2) in eyes:
            # Draw rectangle around each eye
            frame = cv.rectangle(frame, (x + x2, y + y2), (x + x2 + w2, y + y2 + h2), (255, 0, 0), 4)
    
    cv.imshow('Capture - Face detection', frame)

# Command line arguments
parser = argparse.ArgumentParser(description='Code for Cascade Classifier tutorial.')
parser.add_argument('--face_cascade', help='Path to face cascade.', default='algorithm/haarcascade_frontalface_default.xml')
parser.add_argument('--eyes_cascade', help='Path to eyes cascade.', default='algorithm/haarcascade_eye_tree_eyeglasses.xml')
parser.add_argument('--camera', help='Camera device number.', type=int, default=0)
args = parser.parse_args()

face_cascade_name = args.face_cascade
eyes_cascade_name = args.eyes_cascade

face_cascade = cv.CascadeClassifier()
eyes_cascade = cv.CascadeClassifier()

#-- Load the cascades
if not face_cascade.load(face_cascade_name):
    print('--(!)Error loading face cascade')
    exit(0)
if not eyes_cascade.load(eyes_cascade_name):
    print('--(!)Error loading eyes cascade')
    exit(0)

camera_device = args.camera

#-- Read the video stream
cap = cv.VideoCapture(camera_device)
if not cap.isOpened():
    print('--(!)Error opening video capture')
    exit(0)

while True:
    ret, frame = cap.read()
    if frame is None:
        print('--(!) No captured frame -- Break!')
        break

    detectAndDisplay(frame)

    if cv.waitKey(10) == 27:
        break

cap.release()
cv.destroyAllWindows()


# https://docs.opencv.org/3.4/db/d28/tutorial_cascade_classifier.html