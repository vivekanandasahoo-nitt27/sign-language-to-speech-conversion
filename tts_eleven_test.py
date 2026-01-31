import os
from elevenlabs import ElevenLabs

# ✅ Free, default voice (available on free tier)
VOICE_ID = "21m00Tcm4TlvDq8ikWAM"   # example: Rachel's actual ID
OUTPUT_FILE = "elevenlabs_tts_test.mp3"

def test_elevenlabs_tts():
    # 1️⃣ Read API key from environment
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise RuntimeError("❌ ELEVEN_API_KEY not found")

    print("✅ ELEVEN_API_KEY detected")

    # 2️⃣ Create ElevenLabs client
    client = ElevenLabs(api_key=api_key)

    # 3️⃣ Test text
    text = (
        "Hello. This is a successful test of Eleven Labs "
    )

    print("🎙️ Generating speech...")

    # 4️⃣ Generate audio (streaming, safe for containers)
    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_turbo_v2" 
    )

    # 5️⃣ Save audio
    with open(OUTPUT_FILE, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    print(f"✅ Audio generated successfully: {OUTPUT_FILE}")


if __name__ == "__main__":
    test_elevenlabs_tts()
