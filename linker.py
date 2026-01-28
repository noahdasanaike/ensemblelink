"""
Zero-shot record linkage function.
"""

import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm

from .retrieval import EnsembleRetriever
from .reranker import CrossEncoderReranker


def link(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    column_query: str,
    column_corpus: Optional[str] = None,
    retrieval_top_k: int = 20,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
    show_progress: bool = True,
) -> pd.DataFrame:
    """
    Link records from queries to corpus using zero-shot matching.

    Uses ensemble retrieval (dense + sparse) to find candidates, then
    a cross-encoder to score them. All models run locally - no API keys needed.

    Parameters
    ----------
    queries : pd.DataFrame
        The dataset to find matches for.
    corpus : pd.DataFrame
        The reference dataset to match against.
    column_query : str
        Column name in queries containing the text to match.
    column_corpus : str, optional
        Column name in corpus containing the text to match.
        Defaults to column_query if not specified.
    retrieval_top_k : int
        Number of candidates to retrieve per query. Default: 20
    embedding_model : str
        Model for dense retrieval. Default: "Qwen/Qwen3-Embedding-0.6B"
    reranker_model : str
        Model for reranking. Default: "jinaai/jina-reranker-v2-base-multilingual"
    show_progress : bool
        Whether to show progress bars. Default: True

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - query_idx: Index in the query DataFrame
        - query_text: The query text
        - match_idx: Index in the corpus DataFrame
        - match_text: The matched text
        - score: Confidence score

    Example
    -------
    >>> import pandas as pd
    >>> from zeroshot_linkage import link
    >>>
    >>> queries = pd.DataFrame({"name": ["John Smith", "Jane Doe"]})
    >>> corpus = pd.DataFrame({"name": ["J. Smith", "Jane M. Doe", "Bob Wilson"]})
    >>>
    >>> results = link(queries, corpus, column_query="name")
    >>> print(results)
    """
    if column_corpus is None:
        column_corpus = column_query

    # Prepare data
    queries = queries.reset_index(drop=True)
    corpus = corpus.reset_index(drop=True)
    query_texts = queries[column_query].astype(str).tolist()
    corpus_texts = corpus[column_corpus].astype(str).tolist()

    # Build retrieval index
    retriever = EnsembleRetriever(
        embedding_model=embedding_model,
        top_k=retrieval_top_k,
    )
    retriever.index(corpus_texts, show_progress=show_progress)

    # Initialize reranker
    reranker = CrossEncoderReranker(model_name=reranker_model)

    # Link each query
    results = []
    iterator = enumerate(query_texts)
    if show_progress:
        iterator = tqdm(list(iterator), desc="Linking records")

    for query_idx, query_text in iterator:
        # Retrieve candidates
        candidate_indices = retriever.retrieve(query_text)
        candidate_texts = [corpus_texts[i] for i in candidate_indices]

        if not candidate_texts:
            results.append({
                "query_idx": query_idx,
                "query_text": query_text,
                "match_idx": None,
                "match_text": None,
                "score": None,
            })
            continue

        # Rerank candidates
        scores = reranker.score(query_text, candidate_texts)

        # Return best match
        best_local_idx = int(np.argmax(scores))
        best_score = float(scores[best_local_idx])
        match_idx = candidate_indices[best_local_idx]
        match_text = candidate_texts[best_local_idx]

        results.append({
            "query_idx": query_idx,
            "query_text": query_text,
            "match_idx": match_idx,
            "match_text": match_text,
            "score": best_score,
        })

    return pd.DataFrame(results)
