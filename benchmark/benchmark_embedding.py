from typing import List
from openai import OpenAI
import os
from dotenv import load_dotenv
import numpy as np
import mteb
from mteb import MTEB, get_tasks

load_dotenv()
API_KEY = os.getenv("OPENAI_API_KEY")
BASE_URL = "https://mkp-api.fptcloud.com"

MODEL_NAME_1 = "FPT.AI-gte-base"
MODEL_NAME_2 = "Vietnamese_Embedding"

class FPTEmbeddingModel:
    def __init__(self, model_name: str, api_key: str, base_url: str):
        self.model_name = model_name
        self.model_card_data = {"model_id": model_name}
        self.client = OpenAI(api_key=api_key, base_url=base_url)
    def encode(self, sentences: List[str], batch_size: int = 1, **kwargs) -> List[List[float]]:
        embeddings = []
        for sentence in sentences:
            response = self.client.embeddings.create(input=sentence, model=self.model_name)
            print(response.to_dict()["data"][0]["embedding"])
            embeddings.append(response.to_dict()["data"][0]["embedding"])
        return np.array(embeddings)

model1 = FPTEmbeddingModel(
    model_name=MODEL_NAME_1,
    api_key=API_KEY,
    base_url=BASE_URL
)

model2 = FPTEmbeddingModel(
    model_name=MODEL_NAME_2,
    api_key=API_KEY,
    base_url=BASE_URL
)
    
tasks = mteb.get_tasks(tasks=["VieStudentFeedbackClassification"])
evaluation = MTEB(tasks=tasks)
evaluation.run(model1, output_folder=f"benchmark/embedding-results/{MODEL_NAME_1}")
evaluation.run(model2, output_folder=f"benchmark/embedding-results/{MODEL_NAME_2}")

