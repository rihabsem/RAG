import os
from transformers import pipeline
from chromadb import Client
from dotenv import load_dotenv
from transformers import AutoTokenizer, AutoModel
import torch
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
inputs = tokenizer(
    "Hello world! How are you?", 
    return_tensors="pt",  # PyTorch tensors
    truncation=True, #if the vector is longer than the model's max length, truncate it
    padding=True # if the vector is shorter than te model's max length add to it
)
# 3️⃣ Forward pass: feed the inputs (results of the tokenizer) to the model
with torch.no_grad():  #Normally, when training a model, PyTorch tracks every operation to compute gradients for backpropagation. disabled becayse we only need the embedding, not the gradients
    outputs = model(**inputs)

# 4️⃣ Get last hidden state → shape: [batch_size, seq_len, hidden_size]
last_hidden_state = outputs.last_hidden_state #get how each word is represented

print("Last hidden state shape:", last_hidden_state.shape)

# 5️⃣ Mean pooling: average across the token dimension
attention_mask = inputs["attention_mask"]  #defines the part of the vector that are real tokens (1) and the part that are padding (0)

# Expand mask to make sure that it matches the last_hidden_state shape size
mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

masked_embeddings = last_hidden_state * mask # apply the mak to nutralize the padding tokens

sum_embeddings = torch.sum(masked_embeddings, dim=1) # sum the tokens of each vector so v[0] = v[0][0] + v[0][1] + ... + v[0][n] where n is the number of tokens in the vector
# sum_embeddings shape: [batch_size, hidden_size]
sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # counts the number of real tokens in each vector, clamped to avoid division by zero 

sentence_embedding = sum_embeddings / sum_mask

print("Sentence embedding shape:", sentence_embedding.shape)