import time
import cv2
from collections import deque
from typing import List, Optional

import sys
import os

# add project root to path
ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
sys.path.append(ROOT_DIR)

# ⭐ original predictor (UNCHANGED)
from predictor import (extract_landmarks, model, LABELS, CONF_THRESHOLD, STABLE_FRAMES)

from .groq_context import GroqContext
from .memory_service import store_sentence


# ================= CONFIG =================
WORD_GAP = 1.0        # 1 sec → new word
SENTENCE_GAP = 3.0    # 3 sec → sentence end


class PredictorAdapter:
    """
    Streaming segmentation layer:
    letters → words → sentence → NLP → memory
    """

    def __init__(self):
        self.groq = GroqContext()

    # ================= VIDEO PROCESS =================
    def process_video(self, video_path: str, user_id: int) -> List[str]:

        cap = cv2.VideoCapture(video_path)

        pred_queue = deque(maxlen=10)

        stable_candidate = None
        stable_count = 0
        last_added = None

        letters: List[str] = []
        words: List[str] = []
        sentences: List[str] = []

        last_hand_time = time.time()

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            features = extract_landmarks(frame)

            now = time.time()

            # ================= NO HAND =================
            if features is None:
                gap = now - last_hand_time

                # WORD GAP ⭐
                if gap > WORD_GAP and letters:
                    word = "".join(letters)
                    words.append(word)

                    letters.clear()
                    last_added = None
                    stable_candidate = None
                    stable_count = 0

                # SENTENCE GAP ⭐
                if gap > SENTENCE_GAP and words:
                    sentence = self._finalize_sentence(words, user_id)
                    if sentence:
                        sentences.append(sentence)
                    words.clear()

                continue

            # ================= HAND DETECTED =================
            last_hand_time = now

            preds = model.predict(features, verbose=0)
            idx = int(preds.argmax())
            confidence = float(preds.max())

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

        # ================= VIDEO END FINALIZE ⭐ =================

        # letters → word
        if letters:
            words.append("".join(letters))

        # words → sentence
        if words:
            sentence = self._finalize_sentence(words, user_id)
            if sentence:
                sentences.append(sentence)

        return sentences

    # ================= FINALIZE SENTENCE =================
    def _finalize_sentence(self, words: List[str], user_id: int) -> Optional[str]:
        if not words:
            return None

        sentence = self.groq.generate_sentence(user_id, words)

        store_sentence(user_id, sentence)

        return sentence