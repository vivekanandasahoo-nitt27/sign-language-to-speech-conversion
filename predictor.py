import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import joblib
from collections import deque

from groq_llm import GroqLLM
llm = GroqLLM()


# ---------------- CONFIG ----------------
CONF_THRESHOLD = 0.75
STABLE_FRAMES = 5

# ---------------- LOAD MODEL ----------------
model = tf.keras.models.load_model("modelnet_model.h5")

with open("labels.json", "r") as f:
    LABELS = json.load(f)

scaler = joblib.load("scaler.pkl")

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6
)

# ---------------- LANDMARK EXTRACTION (SAME AS TEST FILE) ----------------
def extract_landmarks(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None

    all_landmarks = []

    for hand in result.multi_hand_landmarks[:2]:
        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        wrist = landmarks[0]
        landmarks = landmarks - wrist

        max_dist = np.max(np.linalg.norm(landmarks, axis=1))
        if max_dist == 0:
            return None

        landmarks = landmarks / (max_dist + 1e-6)
        all_landmarks.append(landmarks.flatten())

    if len(all_landmarks) == 1:
        all_landmarks.append(np.zeros(63))

    features = np.concatenate(all_landmarks).reshape(1, -1)
    return scaler.transform(features)

# ---------------- IMAGE PREDICTION ----------------
def predict_image(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return "Invalid image"

    features = extract_landmarks(image)
    if features is None:
        return "No hand detected"

    preds = model.predict(features, verbose=0)
    class_id = int(np.argmax(preds))
    confidence = float(np.max(preds))

    if confidence < CONF_THRESHOLD:
        return "Uncertain"

    return LABELS[class_id]

# ---------------- VIDEO PREDICTION ----------------
def predict_video(video_path):
    cap = cv2.VideoCapture(video_path)

    pred_queue = deque(maxlen=10)
    stable_candidate = None
    stable_count = 0
    last_added = None
    letters = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        features = extract_landmarks(frame)
        if features is None:
            stable_candidate = None
            stable_count = 0
            continue

        preds = model.predict(features, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))

        if confidence < CONF_THRESHOLD:
            continue

        pred_queue.append(idx)
        stable_label = LABELS[max(set(pred_queue), key=pred_queue.count)]

        if stable_label == stable_candidate:
            stable_count += 1
        else:
            stable_candidate = stable_label
            stable_count = 1

        if stable_count >= STABLE_FRAMES and stable_label != last_added:
            letters.append(stable_label)
            last_added = stable_label
            stable_count = 0

    cap.release()

    if not letters:
        return "No signs detected"
    raw_word= "".join(letters)
    try:
        corrected_word = llm.correct_word(letters)
    except Exception as e:
        print("⚠️ NLP error:", e)
        corrected_word = raw_word

    return corrected_word




