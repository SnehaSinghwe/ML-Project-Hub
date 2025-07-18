import cv2
import mediapipe as mp
import numpy as np
import joblib

model = joblib.load("Unified_mentor/new_ASL/asl_mlp_model.pkl")
label_encoder = joblib.load("Unified_mentor/new_ASL/label_encoder.pkl")

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    prediction_text = "Detecting..."

    if result.multi_hand_landmarks:
        for handLms in result.multi_hand_landmarks:
            landmarks = []
            for lm in handLms.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])
            if len(landmarks) == 63:
                prediction = model.predict([landmarks])[0]
                predicted_letter = label_encoder.inverse_transform([prediction])[0]
                prediction_text = f"Detected: {predicted_letter}"

            mp_draw.draw_landmarks(frame, handLms, mp_hands.HAND_CONNECTIONS)

    cv2.putText(frame, prediction_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("ASL Real-Time Detection", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
