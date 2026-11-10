- collections in vector database are what will store the embeddings and any other metadata provided
documents = [
    {"id": "doc1", "text" : "hello world"},
    {"id": "doc2", "text": "how are you today"},
    {"id":"doc3", "text": "goodbye for now"}
]
the vector database stores the id to identify each embedding and sends the content of the "text" is sent
to chromas embedding pipeline to be vectorized then stored in the vectore database

# Explain pooling
from transformers import AutoTokenizer, AutoModel
load_dotenv()

tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
- when using transformers with AutoModel the model doesnt spit out one vector per sentence but a vectore per word
### Example
hello world -> [0.2,0.1] [-0.7, 1.2]
which is not what our vector database expects
to remedy to this problem we use pooling
it essentially means squashing all of those vectors into one vector either by:

Mean pooling → take the average across all token vectors.

Max pooling → take the maximum value for each dimension.

CLS pooling → just use the [CLS] token (special “whole sentence” token in BERT).

### Steps
1) tokenize the sentence:

    tokenizer = AutoTokenizer.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    model = AutoModel.from_pretrained("sentence-transformers/all-MiniLM-L6-v2")
    inputs = tokenizer(
    "Hello world! How are you?", 
    return_tensors="pt",  # PyTorch tensors
    truncation=True, #if the vector is longer than the model's max length, truncate it
    padding=True # if the vector is shorter than te model's max length add to it
)

2) feed the inputs (tokens) to the model

    with torch.no_grad():  #Normally, when training a model, PyTorch tracks every operation to compute gradients for backpropagation. disabled becayse we only need the embedding, not the gradients
    outputs = model(**inputs)
3) get the last hidden state (the shape how each word is represented) and the mask (1 if its a real word 0 if it padding)

    last_hidden_state = outputs.last_hidden_state #get how each word is represented
    print("Last hidden state shape:", last_hidden_state.shape)
    attention_mask = inputs["attention_mask"]  #defines the part of the vector that are real tokens (1) and the part that are padding (0)
4) get rid of the mask tokens by mutiplying the mask and the tokens

    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()

    masked_embeddings = last_hidden_state * mask # apply the mak to nutralize the padding tokens
5) mean pooling
    sum_embeddings = torch.sum(masked_embeddings, dim=1) # sum the tokens of each vector so v[0] = v[0][0] + v[0][1] + ... + v[0][n] where n is the number of tokens in the vector
    #sum_embeddings shape: [batch_size, hidden_size]
    sum_mask = torch.clamp(mask.sum(dim=1), min=1e-9)  # counts the number of real tokens in each vector, clamped to avoid division by zero 

    sentence_embedding = sum_embeddings / sum_mask

