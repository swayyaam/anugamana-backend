import json
import time
import os
import ollama
from tqdm import tqdm

# ---------------- CONFIG ---------------- #
# Make sure you have pulled this model in your terminal first!
# Run: "ollama pull gemma3"
MODEL_NAME = "gemma3:4b" 

def generate_emotions():
    print(f"--- STARTING LOCAL TAGGER (Ollama: {MODEL_NAME}) ---")
    
    # 1. Load Verses
    try:
        with open("gita_full.json", "r", encoding="utf-8") as f:
            verses = json.load(f)
    except FileNotFoundError:
        print("Error: gita_full.json not found.")
        return

    # 2. Resume Capability (Load existing progress)
    if os.path.exists("verse_emotions.json"):
        with open("verse_emotions.json", "r", encoding="utf-8") as f:
            emotion_map = json.load(f)
        print(f"Resuming... Found {len(emotion_map)} existing tags.")
    else:
        emotion_map = {}

    print(f"Processing {len(verses)} verses...")
    
    # 3. Processing Loop
    # We use tqdm for a nice progress bar since local inference takes time per item
    for i, verse in enumerate(tqdm(verses, desc="Tagging Verses")):
        verse_id = verse.get("verse_id")

        # SKIP if already done
        if verse_id in emotion_map and emotion_map[verse_id] != "":
            continue

        # Prepare text (Gemma 2 has a large context window, so we can use more text)
        text_payload = (
            f"Translation: {verse.get('translation', '')}\n"
            f"Purport: {verse.get('purport', '')[:1500]}"
        )

        prompt = (
            "Analyze this text from the Bhagavad Gita. "
            "Identify 3-5 specific human emotions, mental states, or life problems this verse addresses "
            "(e.g. anxiety, grief, duty, confusion, anger, envy, focus). "
            "Return ONLY a comma-separated list of lowercase keywords. Do not write a sentence."
            f"\n\nText:\n{text_payload}"
        )

        try:
            # Call Ollama Locally
            response = ollama.chat(model=MODEL_NAME, messages=[
                {
                    'role': 'user',
                    'content': prompt,
                },
            ])
            
            content = response['message']['content']
            
            # clean up any accidental extra text like "Here are the keywords:"
            clean_keywords = content.replace("Here are the keywords:", "").replace("Keywords:", "").strip()
            
            emotion_map[verse_id] = clean_keywords
            
            # No sleep needed! Local models don't have rate limits.

        except Exception as e:
            print(f"\n❌ Error on {verse_id}: {e}")
            # If Ollama is off, this will crash. Ensure `ollama serve` is running.
            break

        # Periodic Save (every 10 verses)
        if i > 0 and i % 10 == 0:
            with open("verse_emotions.json", "w", encoding="utf-8") as f:
                json.dump(emotion_map, f, indent=2)

    # Final Save
    with open("verse_emotions.json", "w", encoding="utf-8") as f:
        json.dump(emotion_map, f, indent=2)
    
    print("\n--- JOB COMPLETE ---")
    print("Emotions saved to verse_emotions.json")

if __name__ == "__main__":
    generate_emotions()