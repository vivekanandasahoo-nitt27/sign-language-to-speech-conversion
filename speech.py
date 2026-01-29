import pyttsx3
import time

class SpeechEngine:
    def __init__(self, rate=170, volume=1.0, cooldown=2.0):
        self.rate = rate
        self.volume = volume
        self.cooldown = cooldown
        self.last_spoken_text = None
        self.last_spoken_time = 0
        self._init_engine()

    def _init_engine(self):
        self.engine = pyttsx3.init(driverName="sapi5")
        self.engine.setProperty("rate", self.rate)
        self.engine.setProperty("volume", self.volume)

        voices = self.engine.getProperty("voices")
        if voices:
            self.engine.setProperty("voice", voices[0].id)

    def speak(self, text):
        now = time.time()

        if text == self.last_spoken_text and (now - self.last_spoken_time) < self.cooldown:
            return

        self.last_spoken_text = text
        self.last_spoken_time = now

        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except RuntimeError:
            # re-init engine if it silently fails
            self._init_engine()
            self.engine.say(text)
            self.engine.runAndWait()
