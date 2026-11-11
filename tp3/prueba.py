from pathlib import Path
# Install the openai SDK if you haven't already
# !pip install openai

from dotenv import dotenv_values
from openai import OpenAI

api_key = dotenv_values(Path(__file__).with_name(".env")).get("OPENAI_API_KEY")
if not api_key:
    raise SystemExit("No se encontró OPENAI_API_KEY en tp3/.env.")

client = OpenAI(api_key=api_key)

try:
    response = client.chat.completions.create(
        model="gpt-4o-mini", # Using gpt-4o-mini as gpt-5-mini is not currently a valid released model
        messages=[
            {"role": "user", "content": "What is the capital of France?"}
        ]
    )
    print(response.choices[0].message.content)
except Exception as e:
    print(f"An error occurred: {e}")
