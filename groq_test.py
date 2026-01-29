from groq_llm import GroqLLM

llm = GroqLLM()

letters = ["T", "H", "N", "K", "S"]
print(llm.correct_word(letters))
