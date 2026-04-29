import json
import os
import numpy as np
from transformers import AutoTokenizer
from optimum.onnxruntime import ORTModelForFeatureExtraction
from pinecone import Pinecone
from dotenv import load_dotenv

# Load environment variables from .env
load_dotenv()

# Initialize Pinecone and connect to existing index
pc = Pinecone(api_key=os.getenv("PINECONE_API_KEY"))
index = pc.Index("anugamana")

# Load the exact quantized ONNX model used in production
print("Loading quantized ONNX embedding model...")
model_id = "Xenova/all-MiniLM-L6-v2"
tokenizer = AutoTokenizer.from_pretrained(model_id)
model = ORTModelForFeatureExtraction.from_pretrained(
    model_id, subfolder="onnx", file_name="model_quantized.onnx"
)

# Helper function to pool the ONNX outputs into a single 384-dim vector
def get_embedding(text):
    inputs = tokenizer(text, return_tensors="np", padding=True, truncation=True)
    outputs = model(**inputs)
    
    # Mean Pooling
    token_embeddings = outputs.last_hidden_state
    attention_mask = inputs['attention_mask']
    
    input_mask_expanded = np.expand_dims(attention_mask, axis=-1).astype(float)
    input_mask_expanded = np.broadcast_to(input_mask_expanded, token_embeddings.shape)
    
    sum_embeddings = np.sum(token_embeddings * input_mask_expanded, axis=1)
    sum_mask = np.clip(np.sum(input_mask_expanded, axis=1), a_min=1e-9, a_max=None)
    
    pooled_output = sum_embeddings / sum_mask
    return pooled_output[0].tolist()

print("Loading Gita data...")
with open("gita_full.json", "r", encoding="utf-8") as f:
    verses = json.load(f)

print("Indexing verses to Pinecone...")
batch_size = 100
vectors = []

for i, verse in enumerate(verses):
    text_to_embed = f"Chapter {verse['chapter']}, Verse {verse['verse']}: {verse['translation']} {verse.get('purport', '')}"
    
    # Get 384-dim ONNX embedding
    embedding = get_embedding(text_to_embed)
    
    # Same ID format as before so it overwrites smoothly
    vector_id = f"c{verse['chapter']}v{verse['verse']}"
    
    metadata = {
        "chapter": verse['chapter'],
        "verse": verse['verse'],
        "text": verse.get('sanskrit', ''),
        "translation": verse.get('translation', ''),
        "meaning": verse.get('purport', '')
    }
    
    vectors.append({
        "id": vector_id,
        "values": embedding,
        "metadata": metadata
    })
    
    if len(vectors) >= batch_size:
        index.upsert(vectors=vectors)
        vectors = []
        print(f"Upserted batch up to verse {i+1}")

if vectors:
    index.upsert(vectors=vectors)
    print("Final batch upserted.")

print("Indexing complete! Pinecone is now synced with the new ONNX models.")