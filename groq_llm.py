print("✅ USING groq_llm.py FROM:", __file__)
print("✅ GROQ MODEL SET TO:", "llama-3.1-8b-instant")
import os
from groq import Groq


class GroqLLM:
    """
    Lightweight wrapper around Groq LLM
    used for NLP correction of noisy sign-language letters.
    """

    def __init__(self, model="llama-3.1-8b-instant"):
        api_key = os.getenv("GROQ_API_KEY")

        if not api_key:
            raise RuntimeError(
                "❌ GROQ_API_KEY not found. "
                "Set it using environment variables."
            )

        self.client = Groq(api_key=api_key)
        self.model = model

    def correct_word(self, letters):

        if not letters:
            return ""

        raw_word = "".join(letters)

        prompt = f"""
You are correcting noisy letter predictions from a sign language detector.

Input letters: {raw_word}

Return ONLY the most likely correct English word more related and similar to the raw word.
Rules:
- No punctuation
- No explanation
- One word only
"""

        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.2,
            max_tokens=10
        )
        
        corrected = response.choices[0].message.content.strip()
        corrected = corrected.split()[-1]
        return corrected.upper()
