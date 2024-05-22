#!/usr/bin/env python
# coding: utf-8

from tkinter import *
import torch
from torchvision import transforms
import face_recognition
import face_recognition
import cv2
import time
import numpy as np
import os
from PIL import ImageTk, Image 
from ultralytics import YOLO
from torch2trt import TRTModule
import collections
import cv2
import os
import requests
import torch.nn.functional as F


LOADED = False
BASE_PATH = os.path.abspath(os.path.dirname(__file__))

# Classification Models
Base_Classifier = 'models/FireResNet50-100.pt'                            # Initial Model for Classification
Edge_Classifier = 'compressed_models/firenet.pth'                # Compressed Model for Classification

# Detection models 
Base_Detector = 'models/best.pt'                              # Initial Model for Detection                                  
Edge_Detector = 'compressed_models/best.engine' 
Yolo_Detector = "models/yolov8n"    
             # Compressed Model for Detection


BASE_PATH = os.path.abspath(os.path.dirname(__file__))

def load_known_faces_and_names():
    """
    Load known faces and their names from the disk.
    """
    known_face_encodings = []
    known_face_names = []

    # List of tuples containing (image path, name)
    users = [
        ("faces/sidi.jpg", "Sidi"),
        ("faces/seb.jpg", "Seb"),
        ("faces/Antoine.jpg", "Antoine"),
        ("faces/Giovanni_sans_lunette.jpg", "Giovanni"),
        ("faces/Giovanni_lunette.jpg", "Giovanni"),
        ("faces/seb_night.jpg", "Seb"),
        ("faces/Antoine_nuit.jpg", "Antoine"),
        ("faces/Giovanni_night.jpg", "Giovanni")
        
    ]

    for image_path, name in users:
        user_image = face_recognition.load_image_file(image_path)
        user_face_encoding = face_recognition.face_encodings(user_image)[0]
        known_face_encodings.append(user_face_encoding)
        known_face_names.append(name)
    
    return known_face_encodings, known_face_names
                                    

# Slack webhook URL
slack_webhook_url = 'https://hooks.slack.com/services/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'

def send_slack_notification(message):
    payload = {
        "text": message
    }
    response = requests.post(slack_webhook_url, json=payload)
    if response.status_code != 200:
        print(f"Error sending message to Slack: {response.text}")

def authentification(known_face_encodings, known_face_names):
    """
    Fonction qui permet de s'identifier lors du lancement de la Jetson
    """

    # Initialize some variables
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True

    authorized = False
    video_capture = cv2.VideoCapture(0)
    SEUIL = 50
    detected_names = {}
    failed_attempt_count = 0

    while True:
        # Grab a single frame of video
        ret, frame = video_capture.read()

        # Only process every other frame of video to save time
        if process_this_frame:
            small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
            rgb_small_frame = small_frame[:, :, ::-1]
            rgb_small_frame = cv2.cvtColor(rgb_small_frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
            face_names = []

            for face_encoding in face_encodings:
                # See if the face is a match for the known face(s)
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "Unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                    if name in known_face_names:
                        if name not in detected_names:
                            detected_names[name] = 1
                        else:
                            detected_names[name] += 1
                        if detected_names[name] == SEUIL:
                            authorized = True
                            send_slack_notification(f"User '{name}' logged in successfully.")
                            video_capture.release()
                            cv2.destroyAllWindows()
                            return authorized
                    face_names.append(name)
                else:
                    failed_attempt_count += 1
                    if failed_attempt_count == SEUIL:
                        send_slack_notification("Failed login attempt detected.")
                        failed_attempt_count = 0

        process_this_frame = not process_this_frame

        # Display the results
        for (top, right, bottom, left), name in zip(face_locations, face_names):
            # Scale back up face locations since the frame we detected in was scaled to 1/4 size
            top *= 4
            right *= 4
            bottom *= 4
            left *= 4
            # Draw a box around the face
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

            # Draw a label with a name below the face
            cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
            font = cv2.FONT_HERSHEY_DUPLEX
            cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)

        # Display the resulting image
        cv2.imshow("Video", frame)
        # Hit 'q' on the keyboard to quit!
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    # Release handle to the webcam
    video_capture.release()
    cv2.destroyAllWindows()
    return authorized


def fire_detection(classifier_path, detector_path, other_detector_path):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    
    # Load fire model
    imgSize = 224
    fire_model = torch.load(classifier_path)
    
    if isinstance(fire_model, collections.OrderedDict):
        fire_model = TRTModule()
        fire_model.load_state_dict(torch.load(classifier_path))
        
    # Load first YOLO model for fire detection
    yolo_model_fire = YOLO(detector_path, verbose=False)
    
    # Load second YOLO model for other classes
    yolo_model_other = YOLO(other_detector_path, verbose=False)
    
    fire_model.train(False)
    fire_model.to(device)
    
    transform = transforms.Compose([
        transforms.RandomResizedCrop(imgSize),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    
    fire_classes = ["Fire", "No fire", "Start fire"]
    fire_display_classes = ['fire', 'smoke']
    other_display_classes = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat']
    
    average_fps = []
    frame_interval = 2
    notification_sent = False
    
    videos_to_test = ["fire2.mp4"]
        
    for video_to_test in videos_to_test:
        cam_port = os.path.join(BASE_PATH, "videos_tests/" + video_to_test)
        cam = cv2.VideoCapture(cam_port)
        
        prev_frame_time = 0
        new_frame_time = 0
        frame_count = 0
        
        while cam.isOpened():
            success, img = cam.read()
            if not success:
                break
            
            frame_count += 1

            if frame_count % frame_interval != 0:
                continue

            pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            data = transform(pil_img).to(device)
            data = data.unsqueeze(0)

            with torch.no_grad():
                output = fire_model(data)
                classe = output.cpu().numpy()
        
            probabilities = F.softmax(torch.tensor(classe), dim=1).numpy()
            predicted_class = np.argmax(probabilities)

            if predicted_class == 0 or predicted_class == 2:
                results_fire = yolo_model_fire.predict(img, device=device)
                result_fire = results_fire[0]

                boxes_fire = result_fire.boxes.xyxy.cpu().numpy()
                classes_yolo_fire = result_fire.boxes.cls.cpu().numpy()
                confidences_fire = result_fire.boxes.conf.cpu().numpy()
                names_fire = result_fire.names

                for i in range(len(boxes_fire)):
                    box = boxes_fire[i]
                    conf = confidences_fire[i]
                    cls = classes_yolo_fire[i]
                    name = names_fire[int(cls)]

                    if name not in fire_display_classes:
                        continue

                    x1, y1, x2, y2 = map(int, box)
                    color = (255, 128, 0)
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    label = f"{name} {conf:.2f}"
                    cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

                if not notification_sent:
                    if predicted_class == 0:
                        send_slack_notification(f"!!! Fire detected in {video_to_test} at frame {frame_count}!")
                    if predicted_class == 2:
                        send_slack_notification(f"Possible Fire detected in {video_to_test} at frame {frame_count}!")
                    notification_sent = True

            results_other = yolo_model_other.predict(img, device=device)
            result_other = results_other[0]

            boxes_other = result_other.boxes.xyxy.cpu().numpy()
            classes_yolo_other = result_other.boxes.cls.cpu().numpy()
            confidences_other = result_other.boxes.conf.cpu().numpy()
            names_other = result_other.names

            for i in range(len(boxes_other)):
                box = boxes_other[i]
                conf = confidences_other[i]
                cls = classes_yolo_other[i]
                name = names_other[int(cls)]

                if name not in other_display_classes:
                    continue

                x1, y1, x2, y2 = map(int, box)
                color = (0, 255, 0)
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                label = f"{name} {conf:.2f}"
                cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

            cv2.putText(img, fire_classes[predicted_class], (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 0), 3)

            new_frame_time = time.time()
            fps = 1 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)
            average_fps.append(fps)
            cv2.putText(img, "FPS: " + str(fps), (500, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Fire Detection", cv2.resize(img, (640, 480)))

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

        cam.release()
        print(f'Average FPS = {sum(average_fps) / len(average_fps):.2f} FPS')

    cv2.destroyAllWindows()
    del fire_model


def XAI():
    return 



def main():
    # Etape 1: authentification pour accéder à l'interface graphique
    known_face_encodings, known_face_names = load_known_faces_and_names()
    
    # Etape 1: authentification pour accéder à l'interface graphique
    authorized = authentification(known_face_encodings, known_face_names)
    if not authorized:
        print("No authorization")
        from sys import exit
        exit(0)
    

    # Etape 2: interface graphique
    root = Tk()
    root.resizable(width=False, height=False)
    root.geometry("1000x650")
    root.title("Edge AI System")
    
    # Main frame
    main_frame = Frame(root, relief=RIDGE, borderwidth=2)
    main_frame.config(background="green2")
    main_frame.pack(fill=BOTH, expand=1)
    #
    # Welcome message for user
    label_msg = Label(
        main_frame, text=("Welcome!"), bg="green2", font=("Helvetica 24 bold"), height=2
    )
    label_msg.pack(side=TOP)
    label_msg2 = Label(
        main_frame,
        text=("Hello, you are well authorized, congrats !"),
        bg="green2",
        font=("Helvetica 22 bold"),
    )
    label_msg2.pack(side=TOP)
    
    # add logos to the interface
    logo1 = Image.open("logos/all_logos.png")
    logo1 = logo1.resize((990, 90), Image.LANCZOS) 
    logo1 = ImageTk.PhotoImage(logo1)
    logo_label1 = Label(main_frame, image=logo1)
    logo_label1.image = logo1
    logo_label1.pack(side=BOTTOM)

    
    # Ajout texte
    label_msg3 = Label(
        main_frame,
        text=("Initial version"),
        bg="green2",
        fg="black",
        font=("Helvetica 20 bold"),
    )
    label_msg3.place(x=220, y=200)
    label_msg4 = Label(
        main_frame,
        text=("Compressed version"),
        bg="green2",
        fg="black",
        font=("Helvetica 20 bold"),
    )
    label_msg4.place(x=580, y=200)
    # Menu
    but1 = Button(
        main_frame,
        padx=5,
        pady=5,
        width=25,
        height=2,
        bg="white",
        fg="black",
        relief=RAISED,
        command=lambda: fire_detection(Base_Classifier, Base_Detector, Yolo_Detector),
        text="Fire detection",
        font=("helvetica 16 bold"),
    )

    but2 = Button(
        main_frame,
        padx=5,
        pady=5,
        # bd=5,
        height=2,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=XAI,
        text="XAI",
        font=("helvetica 16 bold"),
    )

    but3 = Button(
        main_frame,
        padx=5,
        pady=5,
        height=2,
        # bd=5,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=lambda: fire_detection(Edge_Classifier, Edge_Detector),
        text="Fire detection",
        font=("helvetica 16 bold"),
    )

    but4 = Button(
        main_frame,
        padx=5,
        pady=5,
        height=2,
        # bd=5,
        width=25,
        bg="white",
        fg="black",
        relief=RAISED,
        command=XAI,
        text="XAI",
        font=("helvetica 16 bold"),
    )
    but1.place(x=150, y=280)
    but2.place(x=150, y=380)
    but3.place(x=550, y=280)
    but4.place(x=550, y=380)

    but5 = Button(
        main_frame,
        padx=5,
        pady=5,
        # bd=5,
        height=2,
        width=14,
        bg="white",
        fg="black",
        relief=RAISED,
        command=root.destroy,
        text="Exit",
        font=("helvetica 15 bold"),
    )
    but5.place(x=800, y=480)

    root.mainloop()


if __name__ == "__main__":
    main()
