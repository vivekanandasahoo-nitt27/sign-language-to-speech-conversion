import os
import uuid
from elevenlabs import ElevenLabs

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # example: Rachel's actual ID
OUTPUT_FILE = "elevenlabs_tts_test.mp3"

def text_to_speech(text):
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise RuntimeError("ELEVEN_API_KEY not found")

    client = ElevenLabs(api_key=api_key)

    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_turbo_v2"
    )

    os.makedirs("static/audio", exist_ok=True)
    filename = f"{uuid.uuid4()}.mp3"
    out_path = f"static/audio/{filename}"

    with open(out_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    return out_path
