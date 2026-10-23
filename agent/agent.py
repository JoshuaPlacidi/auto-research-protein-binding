from dotenv import load_dotenv
from google import genai

load_dotenv()

client = genai.Client()

response = client.models.generate_content(
    model="gemini-4-argon",
    contents="Tell me a short joke using your name (and specific version of gemini) as the punchline",
)

print(response.text)