from app.config import TOP_K

def retrieve(vectorstore, query) :
    return vectorstore.similarity_search(query, k = TOP_K)