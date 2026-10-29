import os
from dotenv import load_dotenv
from google import genai

load_dotenv()
client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

skip = ["tts", "lyria", "robotics", "computer-use", "antigravity",
        "deep-research", "image", "live", "audio", "embedding", "veo"]

for m in client.models.list():
    name = m.name
    if "flash" in name and not any(word in name for word in skip):
        print(name)