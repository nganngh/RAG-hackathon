from elasticsearch import Elasticsearch

es = Elasticsearch("http://localhost:9200")

def search_similar(query_vector, index="documents", k=5):
    return es.search()