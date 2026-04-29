# Anugamana — Backend

An AI-powered search API that finds relevant Bhagavad Gita verses based on what you're going through, then uses Gemini to explain how the verse applies to your situation.

## Setup

```bash
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file:
```
ANTHROPIC_API_KEY=your_key_here
```

Build the search index (first run only):
```bash
python indexer.py
```

Start the server:
```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`.
