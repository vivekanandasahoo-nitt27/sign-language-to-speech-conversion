import os
import uuid
from elevenlabs import ElevenLabs

# ⭐ voice config
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
AUDIO_DIR = os.path.join(BASE_DIR, "..", "static", "audio")


def text_to_speech_v2(text: str):
    """
    Convert sentence → speech.
    Safe version (won’t crash pipeline).
    """

    # ---------- safety ----------
    if not text or not text.strip():
        return None

    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        print("⚠️ ELEVEN_API_KEY missing — speech disabled")
        return None

    try:
        client = ElevenLabs(api_key=api_key)

        audio_stream = client.text_to_speech.convert(
            voice_id=VOICE_ID,
            text=text.strip(),
            model_id="eleven_turbo_v2"
        )

        os.makedirs(AUDIO_DIR, exist_ok=True)

        filename = f"{uuid.uuid4()}.mp3"
        file_path = os.path.join(AUDIO_DIR, filename)

        with open(file_path, "wb") as f:
            for chunk in audio_stream:
                f.write(chunk)

        return f"/static/audio/{filename}"

    except Exception as e:
        print("⚠️ Speech generation failed:", e)
        return None