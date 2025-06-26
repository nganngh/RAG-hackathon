from elasticsearch import Elasticsearch

es = Elasticsearch()

def index_document(doc_id, text, vector):
    es.index(index="rag_docs", id=doc_id, document={
        "text": text,
        "embedding": vector
    })


def hybrid_search(query_text, query_vector):
    return es.search(index="rag_docs", body={
        "size": 5,
        "query": {
            "script_score": {
                "query": {
                    "match": { "text": query_text }
                },
                "script": {
                    "source": "cosineSimilarity(params.query_vector, 'embedding') + 1.0",
                    "params": { "query_vector": query_vector }
                }
            }
        }
    })