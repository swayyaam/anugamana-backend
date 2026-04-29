import requests
import json

url = "http://127.0.0.1:8000/search"

# We request limit=1 to trigger the RAG (AI Advice) feature
payload = {
    "query": "I feel lost and confused about my duty.",
    "limit": 1
}

print("Asking the AI...")
response = requests.post(url, json=payload)
data = response.json()

# Check results
if "results" in data and len(data["results"]) > 0:
    top_result = data["results"][0]
    
    print("\n--- 🕉️ VERSE FOUND ---")
    print(f"Verse ID: {top_result['metadata']['verse_id']}")
    print(f"Text: {top_result['text'][:100]}...")
    
    print("\n--- 🤖 AI ADVICE (RAG) ---")
    if "ai_advice" in top_result['metadata']:
        print(f"SUCCESS! \n{top_result['metadata']['ai_advice']}")
    else:
        print("❌ No advice found. Check if your API Key or Model Name is correct.")
else:
    print("❌ No results found.")