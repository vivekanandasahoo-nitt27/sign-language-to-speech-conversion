import os
import uuid
from elevenlabs import ElevenLabs

# ✅ Use valid voice_id (free-tier compatible)
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"

def text_to_speech(text):
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise RuntimeError("❌ ELEVEN_API_KEY not found")

    client = ElevenLabs(api_key=api_key)

    # Generate audio stream
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_turbo_v2"
    )

    # Save audio to static folder
    os.makedirs("static/audio", exist_ok=True)
    filename = f"{uuid.uuid4()}.mp3"
    file_path = os.path.join("static", "audio", filename)

    with open(file_path, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    # ✅ RETURN URL (NOT FILESYSTEM PATH)
    return f"/static/audio/{filename}"
