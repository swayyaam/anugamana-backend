import chromadb
from sentence_transformers import SentenceTransformer

# ---------------- CONFIG ---------------- #
# MUST match what is in main.py and indexer.py
CHROMA_DIR = "chroma_gita"
MODEL_NAME = "sentence-transformers/all-mpnet-base-v2" 
COLLECTION_NAME = "gita_verses"

# ---------------- SETUP ---------------- #
print(f"Loading model: {MODEL_NAME}...")
model = SentenceTransformer(MODEL_NAME)

print(f"Connecting to DB at: {CHROMA_DIR}...")
client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_collection(name=COLLECTION_NAME)

# ---------------- TEST QUERY ---------------- #
query = "I feel afraid and confused about my future"
print(f"\nQuerying: '{query}'")

# 1. Encode
query_embedding = model.encode(query).tolist()

# 2. Search
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3
)

# 3. Print Results (Debug View)
if results['documents'] and results['metadatas']:
    documents = results['documents'][0]
    metadatas = results['metadatas'][0]
    distances = results['distances'][0]

    for i in range(len(documents)):
        print(f"\n--- RESULT {i + 1} (Score: {distances[i]:.4f}) ---")
        print(f"Verse ID : {metadatas[i].get('verse_id')}")
        
        # This is the moment of truth: See if the emotions are there!
        emotions = metadatas[i].get('emotions', 'N/A')
        print(f"Emotions : {emotions}") 
        
        # Show the first 200 chars of text to verify context injection
        print(f"Content  : {documents[i][:200]}...")
else:
    print("No results found.")