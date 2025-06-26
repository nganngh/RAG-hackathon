from pprint import pprint
from openai import OpenAI
from dotenv import load_dotenv
import os 

load_dotenv()

BASE_URL = "https://mkp-api.fptcloud.com"
API_KEY = os.getenv("OPENAI_API_KEY")
MODEL_NAME = 'FPT.AI-gte-base'

client = OpenAI(api_key=API_KEY, 
                base_url=BASE_URL)

def embedding_text(text):
    response = client.embeddings.create(
        input=text,
        model=MODEL_NAME,
    )
    # pprint(response.to_dict())
    return response.to_dict()

vector = embedding_text("Hôm nay tôi buồn!")
print(vector["data"][0]["embedding"])