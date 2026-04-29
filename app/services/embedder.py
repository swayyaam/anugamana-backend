import numpy as np
from app import state


def _mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask_expanded = np.expand_dims(attention_mask, axis=-1)
    summed = np.sum(token_embeddings * mask_expanded, axis=1)
    counts = np.clip(np.sum(mask_expanded, axis=1), a_min=1e-9, a_max=None)
    return summed / counts


def encode_query(text: str) -> list[float]:
    inputs = state.tokenizer_emb(text, padding=True, truncation=True, return_tensors="np")
    outputs = state.embedder(**inputs)
    embedding = _mean_pooling(outputs, inputs["attention_mask"])
    norm = np.linalg.norm(embedding, axis=1, keepdims=True)
    embedding = embedding / np.clip(norm, a_min=1e-9, a_max=None)
    return embedding[0].tolist()


def rerank_pairs(query: str, texts: list[str]) -> list[float]:
    inputs = state.tokenizer_rerank(
        [query] * len(texts), texts,
        padding=True, truncation=True, return_tensors="np",
    )
    outputs = state.reranker(**inputs)
    logits = outputs.logits
    if logits.ndim == 2 and logits.shape[1] == 1:
        return logits[:, 0].tolist()
    return logits.tolist()
