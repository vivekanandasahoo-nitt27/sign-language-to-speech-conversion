import os
from elevenlabs import ElevenLabs

class ElevenTTSEngine:
    def __init__(self, voice="Rachel"):
        api_key = os.getenv("ELEVEN_API_KEY")
        if not api_key:
            raise RuntimeError("❌ ELEVEN_API_KEY not set")

        self.client = ElevenLabs(api_key=api_key)
        self.voice = voice

    def synthesize(self, text, out_path="output.mp3"):
        audio = self.client.text_to_speech.convert(
            voice_id=self.voice,
            text=text,
            model_id="eleven_monolingual_v1"
        )

        with open(out_path, "wb") as f:
            f.write(audio)

        return out_path
