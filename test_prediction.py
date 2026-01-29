import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import joblib
import os
from speech import SpeechEngine   # 🔊 ADD THIS

# ---------------- LOAD MODEL & TOOLS ----------------
MODEL_PATH = "modelnet_model.h5"
LABELS_PATH = "labels.json"
SCALER_PATH = "scaler.pkl"
TEST_IMAGE = "test1.jpg"

print("📌 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

print("📌 Loading labels...")
with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

print("📌 Loading scaler...")
scaler = joblib.load(SCALER_PATH)

# 🔊 Init speech engine
speaker = SpeechEngine(rate=170, volume=1.0, cooldown=2.5)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6
)

# ---------------- LANDMARK EXTRACTION ----------------
def extract_landmarks_from_image(image_path):
    print("📷 Reading image:", image_path)

    if not os.path.exists(image_path):
        print("❌ IMAGE FILE NOT FOUND:", image_path)
        return None

    image = cv2.imread(image_path)
    if image is None:
        print("❌ OpenCV could not read image")
        return None

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)

    if not result.multi_hand_landmarks:
        print("❌ No hands detected")
        return None

    all_landmarks = []

    for hand in result.multi_hand_landmarks[:2]:
        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Normalize relative to wrist
        wrist = landmarks[0]
        landmarks = landmarks - wrist

        # Scale to unit size
        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        landmarks = landmarks / (max_dist + 1e-6)

        all_landmarks.append(landmarks.flatten())

    # Pad second hand if missing
    if len(all_landmarks) == 1:
        all_landmarks.append(np.zeros(63))

    features = np.concatenate(all_landmarks).reshape(1, -1)

    print("📐 Feature shape:", features.shape)

    # Apply saved scaler
    features = scaler.transform(features)

    return features

# ---------------- TEST FUNCTION ----------------
def test_prediction(image_path):
    features = extract_landmarks_from_image(image_path)

    if features is None:
        print("❌ Prediction aborted")
        return

    preds = model.predict(features, verbose=0)

    class_id = np.argmax(preds)
    confidence = np.max(preds)
    label = LABELS[class_id]

    print("\n✅ PREDICTION RESULT")
    print("Label:", label)
    print("Confidence:", round(float(confidence), 4))

    # 🔊 FORCE SPEECH (NO ASYNC, NO CONDITION)
    print("🔊 Speaking now...")
    speaker.speak(str(label))
    print("✅ Speak function finished")


# ---------------- RUN ----------------
if __name__ == "__main__":
    test_prediction(TEST_IMAGE)



    