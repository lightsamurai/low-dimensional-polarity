# encode the 20.000 .json files into .pkl embeddings 

pip install -U sentence-transformers

import pickle
from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

import json

# Our sentences to encode
sentences = [json.loads(line) for line in open(".data/texts/all_bodies_abortion.json", 'r')]
sentences = [comment for sublist in sentences for comment in sublist]


# Sentences are encoded by calling model.encode()
embeddings = model.encode(sentences)


# Store sentences & embeddings on disc
with open("embeddings.pkl", "wb") as fOut:
    pickle.dump({"sentences": sentences, "embeddings": embeddings}, fOut, protocol=pickle.HIGHEST_PROTOCOL)

# Load sentences & embeddings from disc
with open("embeddings.pkl", "rb") as fIn:
    stored_data = pickle.load(fIn)
    stored_sentences = stored_data["sentences"]
    stored_embeddings = stored_data["embeddings"]




