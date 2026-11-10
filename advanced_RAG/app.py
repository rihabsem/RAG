import os
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
from dotenv import load_dotenv
import requests
from pypdf import PdfReader

load_dotenv(dotenv_path= "../.env")
API_key = os.getenv("api_key")
url = "https://api.together.xyz/v1/chat/completions"
headers = {
    "Authorization": f"Bearer {API_key}",
    "Content-Type": "application/json"
}
#steps :
# 1. chunk the data
# 2. embed the data
# 3. store the embeddings in a database

# 1.
reader = PdfReader("data/microsoft-annual-report.pdf")
pdf_text = [page.extract_text().strip() for page in reader.pages] #get rid of white spaves \t and \n -> get a list of the pages
text = [text for text in pdf_text if text] #remove empty strings


from langchain.text_splitter import (
    RecursiveCharacterTextSplitter,
    SentenceTransformersTokenTextSplitter,
)

character_splitter = RecursiveCharacterTextSplitter(
    separators=["\n\n", "\n", ". ", " ", ""], chunk_size=1000, chunk_overlap=0
) #create a splitter that splits the text by paragraphs (\n\n) then by new lines (\n) then by sentences (. ) and finally by words ( ) and characters ("") with a chunk size of 1000 characters and no overlap 
character_split_texts = character_splitter.split_text("\n\n".join(pdf_text)) #join the elements of the pdf_text list with \n\n and apply the splitter

#split text so that the tokens are equal 
sentence_token_splitter = SentenceTransformersTokenTextSplitter(chunk_overlap=0, chunk_size=256)
token_split_text = [sentence_token_splitter.split_text(text) for text in character_split_texts] 

# 2. && 3.
#intialize the database
embedder = SentenceTransformerEmbeddingFunction()
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection_name = "microsoft_annual_report"
collection = chroma_client.get_or_create_collection(
    name=collection_name, 
    embedding_function=embedder
)
for ids, text in enumerate(token_split_text):
    collection.add(
        documents=text, 
        ids=[f"microsoft_annual_report_{ids}_{i}" for i in range(len(text))],
        embeddings=embedder(text)  # embed the text
    )

print(collection.count())  # print the number of documents in the collection

#pass the query through the LLM then contatenate the answer with the query and send it to the chroma database
def augment_query(query):
    data = {
        "model" : "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
            },
            {
                "role": "user",
                "content": f"Question: {query}"
            }

        ],
    }
    return query+" "+requests.post(url, headers=headers, json=data).json()['choices'][0]['message']['content']

query = "What was the total profit for the year, and how does it compare to the previous year?"
augmented_query = augment_query(query)

results = collection.query(
    query_texts=[augmented_query],
    n_results=5,
    include=["documents", "embeddings"]
)

relevent_docs = results["documents"][0]
# Prepare the data for the LLM
query = " ".join(relevent_docs)  # Concatenate the relevant documents into a single string

data = {
        "model" : "mistralai/Mistral-7B-Instruct-v0.2",
        "messages": [
            {
                "role": "system",
                "content": """You are a helpful expert financial research assistant. 
   Provide an example answer to the given question, that might be found in a document like an annual report."""
            },
            {
                "role": "user",
                "content": f"Question: {query}"
            }

        ],
    }
response = requests.post(url, headers=headers, json=data).json()
print("Response:", response['choices'][0]['message']['content'])