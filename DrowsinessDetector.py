
from tkinter import *
import cv2
import mediapipe as mp
import numpy as np
import pygame

# -------------------- SETUP --------------------

mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

pygame.mixer.init()

main = Tk()
main.title("Driver Drowsiness Monitoring")
main.geometry("500x400")
main.config(bg="chocolate1")

running = False

# -------------------- SOUND --------------------

def play_sound(sound):

    if not pygame.mixer.music.get_busy():
        pygame.mixer.music.load(sound)
        pygame.mixer.music.play()

# -------------------- EAR --------------------

def calculate_ear(landmarks, eye):

    v1 = np.linalg.norm(
        np.array(landmarks[eye[1]]) -
        np.array(landmarks[eye[5]])
    )

    v2 = np.linalg.norm(
        np.array(landmarks[eye[2]]) -
        np.array(landmarks[eye[4]])
    )

    h = np.linalg.norm(
        np.array(landmarks[eye[0]]) -
        np.array(landmarks[eye[3]])
    )

    return (v1 + v2) / (2.0 * h)

# -------------------- MAR --------------------

def calculate_mar(landmarks):

    # Upper lip
    top_lip = landmarks[13]

    # Lower lip
    bottom_lip = landmarks[14]

    # Left mouth corner
    left_corner = landmarks[61]

    # Right mouth corner
    right_corner = landmarks[291]

    vertical = np.linalg.norm(
        np.array(top_lip) -
        np.array(bottom_lip)
    )

    horizontal = np.linalg.norm(
        np.array(left_corner) -
        np.array(right_corner)
    )

    mar = vertical / horizontal

    return mar

# -------------------- START --------------------

def start_monitoring():

    global running

    running = True

    status_label.config(text="Monitoring Started")

    cap = cv2.VideoCapture(0)

    EYE_THRESHOLD = 0.25
    EYE_FRAMES = 10

    # Increase/decrease if needed
    MOUTH_THRESHOLD = 0.06

    eye_counter = 0

    left_eye = [33, 160, 158, 133, 153, 144]
    right_eye = [362, 385, 387, 263, 373, 380]

    while running:

        main.update()

        ret, frame = cap.read()

        if not ret:
            break

        frame = cv2.flip(frame, 1)

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = face_mesh.process(rgb)

        if results.multi_face_landmarks:

            for face_landmarks in results.multi_face_landmarks:

                landmarks = [
                    (
                        lm.x * frame.shape[1],
                        lm.y * frame.shape[0]
                    )
                    for lm in face_landmarks.landmark
                ]

                # ---------------- EAR ----------------

                left_ear = calculate_ear(
                    landmarks,
                    left_eye
                )

                right_ear = calculate_ear(
                    landmarks,
                    right_eye
                )

                ear = (left_ear + right_ear) / 2

                # ---------------- MAR ----------------

                mar = calculate_mar(landmarks)

                # ---------------- EYE DETECTION ----------------

                if ear < EYE_THRESHOLD:

                    eye_counter += 1

                    cv2.putText(
                        frame,
                        "Eyes Closed",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 0, 255),
                        2
                    )

                    if eye_counter >= EYE_FRAMES:

                        cv2.putText(
                            frame,
                            "DROWSINESS ALERT!",
                            (20, 80),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.9,
                            (0, 0, 255),
                            3
                        )

                        play_sound("alert.mp3")

                else:

                    eye_counter = 0

                    cv2.putText(
                        frame,
                        "Eyes Open",
                        (20, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (0, 255, 0),
                        2
                    )

                # ---------------- YAWNING DETECTION ----------------

                if mar > MOUTH_THRESHOLD:

                    cv2.putText(
                        frame,
                        "Yawning Detected!",
                        (20, 130),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.9,
                        (255, 0, 0),
                        3
                    )

                    play_sound("alert.mp3")

                # ---------------- VALUES ----------------

                cv2.putText(
                    frame,
                    f"EAR: {ear:.2f}",
                    (320, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    f"MAR: {mar:.2f}",
                    (320, 80),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2
                )

        cv2.imshow(
            "Driver Drowsiness Monitoring",
            frame
        )

        if cv2.waitKey(1) & 0xFF == ord('q'):
            running = False

    cap.release()

    cv2.destroyAllWindows()

    status_label.config(text="Monitoring Stopped")

# -------------------- STOP --------------------

def stop_monitoring():

    global running

    running = False

# -------------------- GUI --------------------

title = Label(
    main,
    text="Driver Drowsiness Monitoring System",
    bg="black",
    fg="white",
    font=("Times", 16, "bold"),
    width=40,
    height=2
)

title.pack(pady=20)

start_button = Button(
    main,
    text="Start Monitoring",
    command=start_monitoring,
    font=("Times", 14, "bold"),
    bg="green",
    fg="white",
    width=18
)

start_button.pack(pady=20)

stop_button = Button(
    main,
    text="Stop Monitoring",
    command=stop_monitoring,
    font=("Times", 14, "bold"),
    bg="red",
    fg="white",
    width=18
)

stop_button.pack(pady=10)

status_label = Label(
    main,
    text="System Ready",
    bg="chocolate1",
    fg="white",
    font=("Times", 13, "bold")
)

status_label.pack(pady=20)

main.mainloop()