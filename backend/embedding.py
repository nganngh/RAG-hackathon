from sentence_transformers import SentenceStransformer

model = SentenceStransformer('all-MiniLM-L6-v2')

def embedding_text(text):
    return model.encode([text])[0]