import os
from elevenlabs import ElevenLabs

client = ElevenLabs(api_key=os.getenv("ELEVEN_API_KEY"))

voices = client.voices.get_all()

print("🎙️ AVAILABLE VOICES:\n")
for v in voices.voices:
    print(f"Name: {v.name}")
    print(f"ID  : {v.voice_id}")
    print("-" * 30)
