import os
import cv2
import numpy as np
import mediapipe as mp
import json

# -------- CONFIG --------
IMAGE_DATA_DIR = "data"
LANDMARK_DIR = "landmark_data"
LABELS_PATH = "labels.json"

os.makedirs(LANDMARK_DIR, exist_ok=True)

mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=2,   # 🔥 SUPPORT 2 HANDS
    min_detection_confidence=0.6
)

skipped = 0
saved = 0

# 🔥 Get labels from folder names
LABELS = sorted([
    d for d in os.listdir(IMAGE_DATA_DIR)
    if os.path.isdir(os.path.join(IMAGE_DATA_DIR, d))
])

print("📌 Detected labels:", LABELS)

for label in LABELS:
    src_folder = os.path.join(IMAGE_DATA_DIR, label)
    dst_folder = os.path.join(LANDMARK_DIR, label)
    os.makedirs(dst_folder, exist_ok=True)

    for idx, img_name in enumerate(os.listdir(src_folder)):
        img_path = os.path.join(src_folder, img_name)

        image = cv2.imread(img_path)
        if image is None:
            skipped += 1
            continue

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        result = hands.process(image_rgb)

        if not result.multi_hand_landmarks:
            skipped += 1
            continue

        all_landmarks = []

        # 🔥 Process up to 2 hands
        for hand in result.multi_hand_landmarks[:2]:
            landmarks = []
            for lm in hand.landmark:
                landmarks.append([lm.x, lm.y, lm.z])

            landmarks = np.array(landmarks)  # (21, 3)

            # 1️⃣ Normalize relative to wrist
            wrist = landmarks[0]
            landmarks = landmarks - wrist

            # 2️⃣ Scale to unit size
            max_dist = np.max(np.linalg.norm(landmarks, axis=1))
            landmarks = landmarks / (max_dist + 1e-6)

            all_landmarks.append(landmarks.flatten())

        # 🔥 If only one hand → pad second hand with zeros
        if len(all_landmarks) == 1:
            all_landmarks.append(np.zeros(63))

        # 🔥 Concatenate → fixed size (126 features)
        final_landmarks = np.concatenate(all_landmarks)

        np.save(
            os.path.join(dst_folder, f"{label}_{idx}.npy"),
            final_landmarks
        )

        saved += 1

# 🔥 SAVE LABELS.JSON
with open(LABELS_PATH, "w") as f:
    json.dump(LABELS, f, indent=2)

print("✅ Landmark extraction completed")
print("📦 Saved files:", saved)
print("❌ Skipped images:", skipped)
print("🏷️ Labels saved to:", LABELS_PATH)
