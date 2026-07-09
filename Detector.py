import cv2
import mediapipe as mp
import numpy as np
import pygame
import threading
import tkinter as tk
from tkinter import Label, Button

# Initialize Mediapipe Face Mesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Initialize Pygame Mixer for Sound Alerts
pygame.mixer.init()

# Tkinter GUI
root = tk.Tk()
root.title("Driver Drowsiness Monitoring System")
root.geometry("500x400")
root.configure(bg="chocolate1")

# Global Variables
running = False  # Control monitoring loop
sound_playing = False  # Track if sound is playing

def play_sound(file_path):
    """Plays alert sound only if not already playing."""
    global sound_playing
    if not sound_playing:  # Prevent multiple sounds
        sound_playing = True
        pygame.mixer.music.load(file_path)
        pygame.mixer.music.play()
        while pygame.mixer.music.get_busy():  # Wait until sound finishes
            if not running:  # Stop sound if monitoring is stopped
                pygame.mixer.music.stop()
                break
        sound_playing = False

def stop_sound():
    """Stops any playing sound immediately."""
    pygame.mixer.music.stop()

def calculate_ear(landmarks, eye_indices):
    """Calculate Eye Aspect Ratio (EAR) for drowsiness detection."""
    A = np.linalg.norm(np.array(landmarks[eye_indices[1]]) - np.array(landmarks[eye_indices[5]]))
    B = np.linalg.norm(np.array(landmarks[eye_indices[2]]) - np.array(landmarks[eye_indices[4]]))
    C = np.linalg.norm(np.array(landmarks[eye_indices[0]]) - np.array(landmarks[eye_indices[3]]))
    return (A + B) / (2.0 * C)

def calculate_mar(landmarks, mouth_indices):
    """Calculate Mouth Aspect Ratio (MAR) for yawning detection."""
    A = np.linalg.norm(np.array(landmarks[mouth_indices[2]]) - np.array(landmarks[mouth_indices[5]]))
    B = np.linalg.norm(np.array(landmarks[mouth_indices[3]]) - np.array(landmarks[mouth_indices[4]]))
    C = np.linalg.norm(np.array(landmarks[mouth_indices[0]]) - np.array(landmarks[mouth_indices[1]]))
    return (A + B) / (2.0 * C)

def start_monitoring():
    """Function to start real-time drowsiness monitoring."""
    global running
    if running:
        return  # Prevent multiple threads from running
    running = True
    status_label.config(text="Webcam Connected Successfully")

    cap = cv2.VideoCapture(0)

    # Drowsiness & Yawn Thresholds
    EYE_AR_THRESH = 0.20
    EYE_AR_CONSEC_FRAMES = 15  # Prevent false alerts
    MOU_AR_THRESH = 0.70

    COUNTER = 0
    yawn_status = False

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                landmarks = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in face_landmarks.landmark]

                # Face landmark indices
                left_eye_indices = [33, 160, 158, 133, 153, 144]
                right_eye_indices = [362, 385, 387, 263, 373, 380]
                mouth_indices = [61, 291, 81, 311, 78, 308, 191]

                # Calculate EAR & MAR
                left_ear = calculate_ear(landmarks, left_eye_indices)
                right_ear = calculate_ear(landmarks, right_eye_indices)
                ear = (left_ear + right_ear) / 2.0
                mar = calculate_mar(landmarks, mouth_indices)

                # Drowsiness Detection
                if ear < EYE_AR_THRESH:
                    COUNTER += 1
                    cv2.putText(frame, "Eyes Closed", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    if COUNTER >= EYE_AR_CONSEC_FRAMES:
                        cv2.putText(frame, "DROWSINESS ALERT!", (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                        threading.Thread(target=play_sound, args=("alert-109578.mp3",)).start()
                else:
                    COUNTER = 0
                    cv2.putText(frame, "Eyes Open", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

                # Yawning Detection
                if mar > MOU_AR_THRESH:
                    cv2.putText(frame, "Yawning Detected!", (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                    if not yawn_status:
                        yawn_status = True
                        threading.Thread(target=play_sound, args=("alert-109578.mp3",)).start()
                else:
                    yawn_status = False

                # Display EAR & MAR
                cv2.putText(frame, f"EAR: {ear:.2f}, MAR: {mar:.2f}", (300, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

        cv2.imshow("Driver Drowsiness Monitoring", frame)

        # Exit condition
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    running = False  # Reset running state when loop exits

def stop_monitoring():
    """Stops the monitoring and stops alert sounds."""
    global running
    running = False
    stop_sound()  # Stop any playing sounds
    status_label.config(text="Monitoring Stopped")

# GUI Components
font = ('times', 16, 'bold')
title = Label(root, text="Driver Drowsiness Monitoring System", bg="black", fg="white", font=font, height=2, width=50)
title.pack(pady=10)

font1 = ('times', 14, 'bold')

start_button = Button(root, text="Start Monitoring", command=lambda: threading.Thread(target=start_monitoring).start(), font=font1, bg="green", fg="white", width=20)
start_button.pack(pady=10)

stop_button = Button(root, text="Stop Monitoring", command=stop_monitoring, font=font1, bg="red", fg="white", width=20)
stop_button.pack(pady=10)

status_label = Label(root, text="", bg="DarkOrange1", fg="white", font=font1, width=40)
status_label.pack(pady=10)

root.mainloop()
