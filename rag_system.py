import os
from sentence_transformers import SentenceTransformer
import chromadb
from dotenv import load_dotenv
import requests
load_dotenv()
API_key = os.getenv("api_key")
url = "https://api.together.xyz/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_key}",
    "Content-Type": "application/json"
}

class MyEmbeddingFunction:
    def __init__(self, model):
        self.model = model

    def __call__(self, input):
        return self.model.encode(input).tolist()

    def name(self):
        return "sentence-transformers"

embedder = SentenceTransformer('all-MiniLM-L6-v2') # embedding model
embedding_function = MyEmbeddingFunction(embedder)

chroma_client = chromadb.PersistentClient(path="chroma_db") #a new repository for the database
collection_name = "my_collection"
collection = chroma_client.get_or_create_collection(name=collection_name, embedding_function = embedding_function)

def load_documents_from_directory(directory_path):
    documents = []
    for filename in os.listdir(directory_path):
        if filename.endswith('txt'):
            with open(
                os.path.join(directory_path, filename), 'r', encoding='utf-8'
            ) as file:
                documents.append({"id" : filename, "text" : file.read()})
    return documents

#split text into chunks
def split_text(text, chunk_size=1000, chunk_overlap = 20):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - chunk_overlap
    return chunks

directory_name = "./news_articles"
documents = load_documents_from_directory(directory_name)
#split documents
chunked_documents = []
for doc in documents:
    chunks = split_text(doc['text'])
    #each document is chunked and the chunks of all the documents are stored in chunked_documents
    for i, chunk in enumerate(chunks):
        chunked_documents.append({"id" : doc['id']+f"chunk{i+1}", "text" : chunk})

#generate embeddings of the chunks in chunked_documents
def get_embedder(text):
    print("Generating embeddings for the text")
    return embedder.encode(text).tolist()

#embedding the chunks in chunked_documents
existing_chunks = collection.get()['ids']
for chunk in chunked_documents:
    chunk['embedding'] = get_embedder(chunk['text'])
    #add the chunks to the collection
    if chunk['id'] in existing_chunks:
        continue
    else:
        collection.add(
            ids = [chunk['id']],
            documents = [chunk['text']],
            embeddings = [chunk['embedding']]
        )

#query the collection
def query_collection(query, n_results=5):
    result = collection.query(
        query_texts=[query],
        n_results=n_results,
        include=["documents", "embeddings"]
    )
    relevant_chunks = [doc for sublist in result['documents'] for doc in sublist]
    return relevant_chunks

#return the response
def generate_response(query, relevant_chunks):
    context = "\n\n".join(relevant_chunks)
    data = {
        "model" : "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {
                "role": "system",
                "content": "You are a helpful assistant that answers questions based on the provided context, if ypu dont know the anwser of the question simply answer with i dont know."
            },
            {
                "role": "user",
                "content": f"Context: {relevant_chunks}\n\nQuestion: {query}"
            }
        ],
    }
    return requests.post(url, headers=headers, json=data).json()
query = " "
chunks = query_collection(query)
result = generate_response(query, chunks)
print("Response:", result['choices'][0]['message']['content'])
    

                







