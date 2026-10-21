import os

import openai
from dotenv import load_dotenv


load_dotenv()


def main() -> None:
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError(
            "No se encontró LLM_API_KEY ni OPENAI_API_KEY en el archivo .env."
        )

    client = openai.OpenAI(
        api_key=api_key,
        base_url=os.getenv(
            "OPENAI_BASE_URL",
            "https://sa-llmproxy.it.itba.edu.ar/",
        ),
    )

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[
            {
                "role": "user",
                "content": "This is a test request. Write a short poem.",
            }
        ],
    )
    print(response.choices[0].message.content)


if __name__ == "__main__":
    main()
