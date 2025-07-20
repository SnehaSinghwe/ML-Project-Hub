import cv2
import mediapipe as mp
import csv
import os
from collections import Counter
import pandas as pd

# --- SETTINGS ---
max_samples_per_emotion = 40
emotions = ['happy', 'sad', 'angry', 'neutral']
key_map = {'h': 'happy', 's': 'sad', 'a': 'angry', 'n': 'neutral'}

# --- SETUP ---
output_file = "emotion_landmarks_dataset.csv"
fieldnames = [f"x{i}" for i in range(468)] + [f"y{i}" for i in range(468)] + ["label"]

# Load existing data or create new file
if not os.path.exists(output_file):
    with open(output_file, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

# Load current label counts
try:
    df = pd.read_csv(output_file)
    label_counts = Counter(df["label"])
except:
    label_counts = Counter()

# MediaPipe setup
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)
cap = cv2.VideoCapture(0)

print("🎥 Starting Data Collection")
print("💡 Press h/s/a/n to label expressions | q to quit")
print("🎯 Target: 40 samples per emotion\n")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = face_mesh.process(frame_rgb)

    if result.multi_face_landmarks:
        landmarks = result.multi_face_landmarks[0].landmark
        x_coords = [lm.x for lm in landmarks]
        y_coords = [lm.y for lm in landmarks]

        # Draw face landmarks
        for lm in landmarks:
            h, w = frame.shape[:2]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        # Display remaining target counts
        y_offset = 30
        for emo in emotions:
            count = label_counts.get(emo, 0)
            cv2.putText(frame, f"{emo}: {count}/{max_samples_per_emotion}",
                        (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 255, 200), 2)
            y_offset += 25

        key = cv2.waitKey(1) & 0xFF
        char = chr(key).lower()

        if char == 'q':
            break

        if char in key_map:
            label = key_map[char]
            if label_counts[label] < max_samples_per_emotion:
                # Save row
                data_row = dict(zip(fieldnames[:-1], x_coords + y_coords))
                data_row["label"] = label
                with open(output_file, "a", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=fieldnames)
                    writer.writerow(data_row)
                label_counts[label] += 1
                print(f"✅ Saved: {label} ({label_counts[label]}/{max_samples_per_emotion})")
            else:
                print(f"⚠️ Already have enough '{label}' samples")

        # Auto stop if all emotions are filled
        if all(label_counts.get(e, 0) >= max_samples_per_emotion for e in emotions):
            print("\n🎉 All emotion targets reached! Collection complete.")
            break

    cv2.imshow("Emotion Data Collector", frame)

cap.release()
cv2.destroyAllWindows()
