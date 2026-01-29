import cv2
import numpy as np
import mediapipe as mp
import tensorflow as tf
import json
import joblib
import time
from collections import deque

from speech import SpeechEngine
from groq_llm import GroqLLM


# ================== CONFIG ==================
MODEL_PATH = "modelnet_model.h5"
LABELS_PATH = "labels.json"
SCALER_PATH = "scaler.pkl"

CAPTURE_DURATION = 10      # seconds
CONF_THRESHOLD = 0.7
STABLE_WINDOW = 6          # frames for stability


# ================== LOAD MODEL ==================
print("📌 Loading model...")
model = tf.keras.models.load_model(MODEL_PATH)

with open(LABELS_PATH, "r") as f:
    LABELS = json.load(f)

scaler = joblib.load(SCALER_PATH)

speaker = SpeechEngine(rate=170, volume=1.0, cooldown=1.5)
llm = GroqLLM()


# ================== MEDIAPIPE ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.6
    
)
mp_draw = mp.solutions.drawing_utils


# ================== FEATURE EXTRACTION ==================
def extract_features_from_frame(frame):
    """
    Extracts 126-D landmark features from up to 2 hands.
    Hand order is fixed left -> right.
    """
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    if not result.multi_hand_landmarks:
        return None, result

    hands_landmarks = []

    for hand in result.multi_hand_landmarks:
        lm = np.array([[p.x, p.y, p.z] for p in hand.landmark])

        # normalize relative to wrist
        wrist = lm[0]
        lm = lm - wrist

        max_norm = np.linalg.norm(lm, axis=1).max()
        if max_norm == 0:
            return None, result

        lm = lm / (max_norm + 1e-6)
        hands_landmarks.append(lm)

    # 🔥 IMPORTANT: sort hands left -> right
    hands_landmarks.sort(key=lambda x: x[0][0])

    # pad if only one hand
    if len(hands_landmarks) == 1:
        hands_landmarks.append(np.zeros((21, 3)))

    features = np.concatenate(
        [h.flatten() for h in hands_landmarks[:2]]
    ).reshape(1, -1)

    features = scaler.transform(features)
    return features, result


# ================== LETTER EXTRACTION (10s) ==================
def extract_letters_10s(duration=CAPTURE_DURATION):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Webcam not accessible")
        return []

    start_time = time.time()
    pred_queue = deque(maxlen=STABLE_WINDOW)
    letter_buffer = []
    last_added = None

    print(f"🎥 Capturing for {duration} seconds...")

    while time.time() - start_time < duration:
        ret, frame = cap.read()
        if not ret:
            break

        

        features, result = extract_features_from_frame(frame)

        if features is None:
            pred_queue.clear()
            continue

        preds = model.predict(features, verbose=0)
        idx = int(np.argmax(preds))
        confidence = float(np.max(preds))
        label = LABELS[idx]

        if not label.isalpha():
            continue

        pred_queue.append(label)
        stable_label = max(set(pred_queue), key=pred_queue.count)

        if confidence > CONF_THRESHOLD and stable_label != last_added:
            letter_buffer.append(stable_label)
            last_added = stable_label
            print("📌 Letters:", "".join(letter_buffer))

        # --------- DISPLAY (optional) ----------
        cv2.putText(frame, stable_label, (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1,
                    (0, 255, 0), 2)

        if result and result.multi_hand_landmarks:
            for handLms in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(
                    frame, handLms, mp_hands.HAND_CONNECTIONS
                )

        cv2.imshow("Live ASL Capture (10s)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()
    return letter_buffer


# ================== MAIN PIPELINE ==================
def run_live_sign_to_speech():
    letters = extract_letters_10s()

    if not letters:
        print("❌ No letters detected")
        return

    raw_word = "".join(letters)
    print("\n🧠 Raw letters:", raw_word)

    corrected_word = llm.correct_word(letters)
    print("🤖 NLP corrected word:", corrected_word)

    speaker.speak(corrected_word)


# ================== RUN ==================
if __name__ == "__main__":
    run_live_sign_to_speech()
