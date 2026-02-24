from typing import List
from groq import Groq
import os

from .memory_service import get_context, build_system_prompt


class GroqContext:
    

    def __init__(self, model: str = "llama-3.1-8b-instant"):
        self.model = model

        api_key = os.getenv("GROQ_API_KEY")
        if not api_key:
            print("⚠️ GROQ_API_KEY missing — NLP disabled")
            self.client = None
            return

        self.client = Groq(api_key=api_key)

    # ================= GENERATE SENTENCE =================
    def generate_sentence(self, user_id: int, words: List[str]) -> str:

        if not words:
            return ""

        # ⭐ fetch memory context
        history = get_context(user_id)

        # ⭐ system prompt with context rules
        system_prompt = build_system_prompt(history)

        # ⭐ words → user prompt (minimal → stable output)
        words_text = " ".join(words)

        user_prompt = f"""
Words:
{words_text}

Return only the sentence.
"""

        # ⭐ fallback if Groq disabled
        if self.client is None:
            return words_text

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.2,   # ⭐ lower = more deterministic
                max_tokens=40
            )

            sentence = response.choices[0].message.content.strip()

            # ⭐ safety cleanup (very important)
            sentence = self._cleanup(sentence)

            return sentence

        except Exception as e:
            print("⚠️ Groq context generation failed:", e)
            return words_text

    # ================= CLEANUP OUTPUT =================
    def _cleanup(self, text: str) -> str:
        """
        Ensures sentence-only output.
        Removes assistant style artifacts.
        """

        if not text:
            return ""

        # remove quotes
        text = text.strip().strip('"').strip("'")

        # remove common assistant prefixes
        prefixes = [
            "Sentence:",
            "Answer:",
            "Output:",
        ]

        for p in prefixes:
            if text.lower().startswith(p.lower()):
                text = text[len(p):].strip()

        return text