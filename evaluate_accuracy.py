import requests
import numpy as np
import time

# ---------------- CONFIG ---------------- #
API_URL = "http://127.0.0.1:8000/search"

# The "Golden Dataset": Queries and their expected Verse ID (Ground Truth)
# You can expand this list for your assignment.
GOLDEN_DATASET = [
    ("What is the duty of a warrior?", "2.31"),
    ("You have a right to perform your prescribed duty", "2.47"),
    ("Why should I not grieve for the dead?", "2.27"),
    ("anger leads to delusion", "2.63"),
    ("peace attained by abandoning desires", "2.71"),
    ("practice of yoga", "6.12"),
    ("I am the source of all spiritual and material worlds", "10.8"),
    ("divine eye to see the cosmic form", "11.8"),
    ("abandon all varieties of religion", "18.66"),
    ("whenever there is a decline in religion", "4.7")
]

def evaluate():
    print(f"Starting Evaluation on {len(GOLDEN_DATASET)} queries...\n")
    
    reciprocal_ranks = []
    hits_at_1 = 0
    hits_at_5 = 0
    
    for query, expected_id in GOLDEN_DATASET:
        try:
            # Call your API
            response = requests.post(API_URL, json={"query": query, "limit": 10})
            results = response.json().get("results", [])
            
            # Extract IDs from results
            found_ids = [r["metadata"]["verse_id"] for r in results]
            
            # Calculate Rank
            rank = float('inf')
            if expected_id in found_ids:
                rank = found_ids.index(expected_id) + 1 # 1-based index
            
            # Calculate Metrics
            rr = 1.0 / rank if rank != float('inf') else 0.0
            reciprocal_ranks.append(rr)
            
            if rank == 1: hits_at_1 += 1
            if rank <= 5: hits_at_5 += 1
            
            # Log specific misses for debugging
            status = "✅" if rank <= 5 else "❌"
            print(f"{status} Query: '{query}' | Expected: {expected_id} | Found at Rank: {rank if rank < 100 else 'Not Found'}")
            
        except Exception as e:
            print(f"Error evaluating '{query}': {e}")

    # Final Metrics
    mrr = np.mean(reciprocal_ranks)
    recall_at_5 = hits_at_5 / len(GOLDEN_DATASET)
    
    print("\n" + "="*30)
    print("       EVALUATION REPORT       ")
    print("="*30)
    print(f"Total Queries: {len(GOLDEN_DATASET)}")
    print(f"MRR (Mean Reciprocal Rank): {mrr:.4f}")
    print(f"Recall@5 (Accuracy):        {recall_at_5 * 100:.1f}%")
    print("="*30)

if __name__ == "__main__":
    evaluate()