"""
recommender.py
---------------
Recommends tutorials/quizzes/resources for a passage, WITHOUT calling
a generative model -- this is retrieval, not generation, so it stays
fast and cheap:

    passage -> keywords (KeyBERT)
            -> embed keywords (sentence-transformers)
            -> nearest-neighbor search over a curated resource
               database (embedded once, reused every request)

RESOURCE_DB below is a placeholder. Replace it with your own curated
list -- topic, title, url, type (tutorial/quiz/video/docs) -- for
whatever domain your notes cover. This is the piece where domain
curation matters more than model choice.
"""

from functools import lru_cache
from typing import List, Dict

# ---------------------------------------------------------------------
# 1. Curated resource database (placeholder -- expand this)
# ---------------------------------------------------------------------

RESOURCE_DB: List[Dict[str, str]] = [
    {
        "topic": "supervised learning classification regression",
        "title": "Supervised Learning Crash Course",
        "url": "https://www.khanacademy.org/computing/computer-science/algorithms",
        "type": "tutorial",
    },
    {
        "topic": "overfitting regularization cross-validation",
        "title": "Overfitting and Regularization Explained",
        "url": "https://developers.google.com/machine-learning/crash-course/overfitting",
        "type": "tutorial",
    },
    {
        "topic": "decision trees random forest",
        "title": "Decision Trees Quiz",
        "url": "https://www.geeksforgeeks.org/decision-tree/",
        "type": "quiz",
    },
    {
        "topic": "neural networks backpropagation",
        "title": "Neural Networks and Deep Learning (free online book)",
        "url": "http://neuralnetworksanddeeplearning.com/",
        "type": "tutorial",
    },
    # ... add more, ideally dozens per domain you expect notes to cover
]


# ---------------------------------------------------------------------
# 2. Lazy-loaded models (keyword extractor + embedder)
# ---------------------------------------------------------------------

@lru_cache(maxsize=1)
def _get_keyword_model():
    from keybert import KeyBERT

    return KeyBERT(model="all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_embedder():
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer("all-MiniLM-L6-v2")


@lru_cache(maxsize=1)
def _get_resource_index():
    """
    Embed every resource's topic string once and cache it, along with a
    FAISS index for fast nearest-neighbor lookup at query time.
    """
    import faiss
    import numpy as np

    embedder = _get_embedder()
    topic_strings = [r["topic"] for r in RESOURCE_DB]
    embeddings = embedder.encode(topic_strings, normalize_embeddings=True)

    index = faiss.IndexFlatIP(embeddings.shape[1])  # cosine sim via normalized dot product
    index.add(np.array(embeddings, dtype="float32"))
    return index


# ---------------------------------------------------------------------
# 3. Public function
# ---------------------------------------------------------------------

def recommend_resources(passage: str, top_k: int = 3) -> List[Dict[str, str]]:
    """
    Extract keywords from the passage, then return the top_k closest
    resources from RESOURCE_DB by semantic similarity.
    """
    import numpy as np

    kw_model = _get_keyword_model()
    keywords = kw_model.extract_keywords(
        passage, keyphrase_ngram_range=(1, 2), stop_words="english", top_n=5
    )
    query = " ".join(k for k, _score in keywords)

    embedder = _get_embedder()
    query_vec = embedder.encode([query], normalize_embeddings=True)

    index = _get_resource_index()
    scores, indices = index.search(np.array(query_vec, dtype="float32"), top_k)

    results = []
    for idx, score in zip(indices[0], scores[0]):
        if idx == -1:
            continue
        resource = dict(RESOURCE_DB[idx])
        resource["relevance"] = float(score)
        results.append(resource)

    return results


if __name__ == "__main__":
    sample = (
        "Overfitting occurs when a model learns the training data too well "
        "and fails to generalize to new data."
    )
    for r in recommend_resources(sample):
        print(f"[{r['type']}] {r['title']} ({r['relevance']:.2f}) -> {r['url']}")