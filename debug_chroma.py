import chromadb
from chromadb.config import Settings

client = chromadb.PersistentClient(path="chroma_gita")

print("Collections found:")
for c in client.list_collections():
    print("-", c.name)

print("Collections found:")
for c in client.list_collections():
    print("-", c.name)
