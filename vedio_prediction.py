import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import joblib
from collections import deque

from speech import SpeechEngine
from groq_llm import GroqLLM

# ---------------- CONFIG ----------------
VIDEO_PATH = "input_video.mp4"   # path to uploaded video
CONF_THRESHOLD = 0.75
STABLE_FRAMES = 5   # number of consecutive frames required

# ---------------- LOAD MODEL ----------------
print("📌 Loading model...")
model = tf.keras.models.load_model("modelnet_model.h5")

with open("labels.json", "r") as f:
    LABELS = json.load(f)

scaler = joblib.load("scaler.pkl")

# ---------------- NLP + SPEECH ----------------
speaker = SpeechEngine(rate=170, volume=1.0, cooldown=1.0)
llm = GroqLLM()

# ---------------- MEDIAPIPE ----------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,
    min_detection_confidence=0.6,
    
    
    
)

# ---------------- BUFFERS ----------------
detected_letters = []  
frame_count = 0
letter_buffer = []
pred_queue = deque(maxlen=10)

last_added = None
stable_candidate = None
stable_count = 0

# ---------------- FEATURE EXTRACTION ----------------
def extract_features(frame):
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)
    except Exception:
        return None

    if not result.multi_hand_landmarks:
        return None

    all_landmarks = []

    # Take up to 2 hands
    for hand in result.multi_hand_landmarks[:2]:
        landmarks = []
        for lm in hand.landmark:
            landmarks.append([lm.x, lm.y, lm.z])

        landmarks = np.array(landmarks)

        # Normalize relative to wrist
        wrist = landmarks[0]
        landmarks = landmarks - wrist

        max_norm = np.linalg.norm(landmarks, axis=1).max()
        if max_norm == 0:
            return None

        landmarks = landmarks / (max_norm + 1e-6)

        all_landmarks.append(landmarks.flatten())

    
    if len(all_landmarks) == 1:
        all_landmarks.append(np.zeros(63))

    features = np.concatenate(all_landmarks).reshape(1, -1)

    # Now features.shape == (1, 126)
    features = scaler.transform(features)

    return features

# ---------------- MAIN FUNCTION ----------------
def predict_from_video(video_path):
    global last_added, stable_candidate, stable_count
    frame_count = 0 

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("❌ Cannot open video:", video_path)
        return

    print("🎥 Processing video:", video_path)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        
        features = extract_features(frame)

        if features is None:
            # reset stability when hand disappears
            stable_candidate = None
            stable_count = 0
            continue

        frame_count += 1

        preds = model.predict(features, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = LABELS[idx]

        # ✅ Store detected letter (for checking)
        if confidence > CONF_THRESHOLD:
            detected_letters.append(label)

        # 🧪 Print array every 20 frames
        if frame_count % 20 == 0:
            print("🧪 Detected letters so far:", detected_letters)


        pred_queue.append(idx)
        stable_label = LABELS[max(set(pred_queue), key=pred_queue.count)]

        if confidence > CONF_THRESHOLD:
            if stable_label == stable_candidate:
                stable_count += 1
            else:
                stable_candidate = stable_label
                stable_count = 1

            if stable_count >= STABLE_FRAMES and stable_label != last_added:
                letter_buffer.append(stable_label)
                last_added = stable_label
                stable_count = 0

                print("📌 Letters:", "".join(letter_buffer))

    cap.release()

    # ---------------- NLP + SPEECH ----------------
    if not letter_buffer:
        print("❌ No letters detected in video")
        return

    raw_word = "".join(letter_buffer)
    print("\n🧠 Raw letters:", raw_word)

    try:
        corrected_word = llm.correct_word(letter_buffer)
    except Exception as e:
        print("⚠️ Groq error:", e)
        corrected_word = raw_word

    print("🤖 NLP corrected word:", corrected_word)
    speaker.speak(corrected_word)

# ---------------- RUN ----------------
if __name__ == "__main__":
    try:
        predict_from_video(VIDEO_PATH)
    except Exception as e:
        print("❌ Fatal error:", repr(e))

