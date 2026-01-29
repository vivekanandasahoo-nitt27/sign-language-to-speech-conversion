import os
from elevenlabs import ElevenLabs

VOICE_ID = "21m00Tcm4TlvDq8ikWAM"


def test_tts():
    api_key = os.getenv("ELEVEN_API_KEY")
    if not api_key:
        raise RuntimeError("❌ ELEVEN_API_KEY not found")

    print("✅ ELEVEN_API_KEY found")

    client = ElevenLabs(api_key=api_key)

    text = "Hello, this is a test of Eleven Labs text to speech."

    print("🎙️ Generating speech...")

    audio_stream = client.text_to_speech.convert(
        voice_id=VOICE_ID,
        text=text,
        model_id="eleven_monolingual_v1"
    )

    out_file = "tts_test_output.mp3"
    with open(out_file, "wb") as f:
        for chunk in audio_stream:
            f.write(chunk)

    print(f"✅ Audio saved as {out_file}")

if __name__ == "__main__":
    test_tts()
