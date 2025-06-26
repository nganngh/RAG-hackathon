import openai
import os
from dotenv import load_dotenv

load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

def call_llm(prompt):
    response = openai.ChatCompletion.create(
        model="gpt-4",
        messages=[
            {"role": "system", "content": "you are a helpful assistant"},
            {"role": "user", "content": prompt}
        ]
    )
    return response.choices[0].message["content"]