from typing import List
from .db import insert_sentence, get_last_sentences


# ================= STORE SENTENCE =================
def store_sentence(user_id: int, sentence: str):
    """
    Save final generated sentence to DB.
    """
    if not sentence or not sentence.strip():
        return

    insert_sentence(user_id, sentence.strip())


# ================= FETCH CONTEXT =================
def get_context(user_id: int, limit: int = 3) -> List[str]:
    """
    Get last N sentences for conversational context.
    Returned oldest -> newest (already handled in db layer).
    """
    if not user_id:
        return []

    history = get_last_sentences(user_id, limit)

    # safety clean
    history = [s.strip() for s in history if s and s.strip()]

    return history


# ================= BUILD SYSTEM PROMPT ⭐ CORE =================
def build_system_prompt(history: List[str]) -> str:
    """
    Convert history into system prompt for Groq.
    This is the key for conversational NLP.
    """

    if not history:
        return (
            "You convert sign language words into natural spoken English sentences. "
            "Keep sentences short and clear."
        )

    history_text = "\n".join(f"- {s}" for s in history)

    system_prompt = f"""
You convert sign language words into natural spoken English sentences.

Conversation history:
{history_text}

Keep consistency with previous conversation.
Generate short natural sentences.
"""

    return system_prompt.strip()