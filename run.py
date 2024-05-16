import os
import cv2
import dlib
import numpy as np
import face_recognition
import time
import pyttsx3

engine = pyttsx3.init()
face_algorithm = "algorithm/haarcascade_frontalface_default.xml"

def speak(name):
    engine.say(name)
    engine.runAndWait()

def recognize_faces_in_video(video_capture, known_encodings, known_names, folder_path, unknown_faces_folder):
    face_cascade = cv2.CascadeClassifier(face_algorithm)
    recognized_names = set()
    unknown_face_captured = False
    last_spoken_time = {}
    delay_seconds = 10

    while True:
        ret, frame = video_capture.read()
        if not ret:
            break

        frame = cv2.resize(frame, (640, 480))
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)

        for (x, y, w, h) in faces:
            face_region = frame[y:y+h, x:x+w]
            rgb_face_region = cv2.cvtColor(face_region, cv2.COLOR_BGR2RGB)

            face_encodings = face_recognition.face_encodings(rgb_face_region)

            if len(face_encodings) > 0:
                face_encoding = face_encodings[0]
                distances = face_recognition.face_distance(known_encodings, face_encoding)
                min_distance_index = np.argmin(distances)
                min_distance = distances[min_distance_index]

                current_time = time.time()

                if min_distance < 0.6:
                    name = known_names[min_distance_index]
                    if name not in last_spoken_time or (current_time - last_spoken_time[name] > delay_seconds):
                        speak(name) 
                        last_spoken_time[name] = current_time
                    recognized_names.add(name)
                    unknown_face_captured = False 
                else:
                    if not unknown_face_captured:
                        unknown_face_path = os.path.join(unknown_faces_folder, f"unknown_face_{time.time()}.jpg")
                        cv2.imwrite(unknown_face_path, face_region)
                        unknown_face_captured = True

                    if "Unknown" not in last_spoken_time or (current_time - last_spoken_time["Unknown"] > delay_seconds):
                        speak("Unknown")
                        last_spoken_time["Unknown"] = current_time
                    name = "Unknown"

                if name != "Unknown":
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.rectangle(frame, (x, y + h - 30), (x + w, y + h), (0, 255, 0), cv2.FILLED)
                    font = cv2.FONT_HERSHEY_DUPLEX
                    cv2.putText(frame, name, (x + 6, y + h - 6), font, 0.7, (0, 0, 0), 1)

        cv2.imshow('Video', frame)

        if cv2.waitKey(10) == 27:
            break

        if time.time() % 3 == 0:
            known_encodings, known_names = load_known_faces(folder_path)

def load_known_faces(folder_path):
    known_encodings = []
    known_names = []
    image_files = []
    
    for file in os.listdir(folder_path): 
        if file.endswith(('.jpg', '.png', '.jpeg')):
            image_files.append(file)
        
    for image_file in image_files:
        image_path = os.path.join(folder_path, image_file)
        name = os.path.splitext(image_file)[0]
        image = face_recognition.load_image_file(image_path)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(name)

    return np.array(known_encodings), np.array(known_names)

folder_path = "faces"
known_encodings, known_names = load_known_faces(folder_path)

unknown_faces_folder = "unknown_faces"
os.makedirs(unknown_faces_folder, exist_ok=True)

print("Surce:")
print("1. Laptop Camera")
print("2. Mobile Camera")
print("3. Other Camera")
choice = input("Enter 1, 2, or 3: ")

if choice == '1':
    video_source = 0
elif choice == '2':
    video_source = "http://192.168.43.1:8080/video"  
elif choice == '3':
    other_camera_index = int(input("Enter the camera index :"))
    video_source = other_camera_index
else:
    print("Invalid choice. Defaulting to laptop camera.")
    video_source = 0

video_capture = cv2.VideoCapture(video_source)

recognize_faces_in_video(video_capture, known_encodings, known_names, folder_path, unknown_faces_folder)

video_capture.release()
cv2.destroyAllWindows()
