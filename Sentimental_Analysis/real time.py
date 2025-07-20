import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load model and encoder
model = joblib.load("emotion_model.pkl")
le = joblib.load("label_encoder.pkl")

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

cap = cv2.VideoCapture(0)
print("🎥 Real-Time Emotion Detection — Press Q to quit")

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

        features = np.array(x_coords + y_coords).reshape(1, -1)
        prediction = model.predict(features)
        label = le.inverse_transform(prediction)[0]

        for lm in landmarks:
            h, w = frame.shape[:2]
            cx, cy = int(lm.x * w), int(lm.y * h)
            cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

        cv2.putText(frame, f"Emotion: {label}", (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (100, 255, 100), 3)

    cv2.imshow("Real-Time Emotion Detector", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
