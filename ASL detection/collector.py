import cv2
import mediapipe as mp
import numpy as np
import pandas as pd
import os
import string

# ====== CONFIGURATION ======
SAMPLES_PER_LETTER = 200
OUTPUT_DIR = "asl_landmark_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ====== SETUP ======
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# ====== MAIN LOOP ======
print("Initializing webcam...")
cap = cv2.VideoCapture(0)
print("[INFO] Starting data collection.")
print("[INFO] Press a letter key (A-Z) to start collecting for that letter.")
print("[INFO] Press 'ESC' to exit.")

current_label = None
sample_count = 0
data = []

print("Before entering while loop...")
while True:
    print("looping...")
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to read from webcam.")
        break


    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    try:
        result = hands.process(rgb)
    except Exception as e:
        print("[ERROR] Mediapipe processing failed:", e)
        break


    if result.multi_hand_landmarks and current_label is not None and sample_count < SAMPLES_PER_LETTER:
        for handLms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            data.append([current_label] + landmarks)
            sample_count += 1
            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    # Show collection info
    if current_label:
        status = f"Collecting for '{current_label}' ({sample_count}/{SAMPLES_PER_LETTER})"
    else:
        status = "Press a letter key (A-Z) to start..."

    cv2.putText(frame, status, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
    cv2.imshow("ASL Landmark Collector", frame)

    key = cv2.waitKey(1) & 0xFF

    # If ESC is pressed, exit
    if key == 27:
        break

    # If letter key pressed, start collecting for that letter
    if 65 <= key <= 90 or 97 <= key <= 122:  # A-Z or a-z
        current_label = chr(key).upper()
        print(f"\n[INFO] Starting collection for '{current_label}'")
        sample_count = 0
        data = []

    # If enough samples collected, save automatically
    if sample_count == SAMPLES_PER_LETTER and current_label:
        df = pd.DataFrame(data)
        filename = os.path.join(OUTPUT_DIR, f"landmarks_{current_label}.csv")
        df.to_csv(filename, index=False, header=False)
        print(f"[SAVED] {filename} ({sample_count} samples)")
        current_label = None
        sample_count = 0
        data = []

cap.release()
cv2.destroyAllWindows()
print("[INFO] Data collection completed.")

