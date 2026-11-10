import chromadb
chroma_client = chromadb.Client()

#Collections are where you'll store your embeddings, documents, and any additional metadata. 
#Collections index your embeddings and documents, and enable efficient retrieval and filtering

collection_name = "my_collection"
collection = chroma_client.create_collection(name=collection_name)

documents = [
    {"id": "doc1", "text" : "hello world"},
    {"id": "doc2", "text": "how are you today"},
    {"id":"doc3", "text": "goodbye for now"},
    {"id":"doc3", "text": "today we are serving grilled cheese sandwiches"}
]
for document in documents:
    collection.upsert(
        ids = document["id"],
        documents = document["text"]
    )
query = "im hungry"
results = collection.query(
    query_texts=[query],
    n_results=3,  # Number of results to return
)
res = results['documents'][0][0]
if results["distances"][0][0] > 0.5:
    print("No match found")
else:
    print(f"Found a match: {res}")