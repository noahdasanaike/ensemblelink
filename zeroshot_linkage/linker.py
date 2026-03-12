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
    column_query: str = None,
    column_corpus: Optional[str] = None,
    columns_query: Optional[list] = None,
    columns_corpus: Optional[list] = None,
    retrieval_top_k: int = 20,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    batch_size: int = 50000,
) -> pd.DataFrame:
    """
    Link records from queries to corpus using zero-shot matching.

    Uses ensemble retrieval (dense + sparse) to find candidates, then
    a cross-encoder to score them. All models run locally - no API keys needed.

    For multi-column matching, pass ``columns_query`` (and optionally
    ``columns_corpus``) instead of ``column_query``. The columns are
    concatenated with " | " separators before matching. Concatenation
    consistently outperforms blocking-based approaches across benchmarks
    (see Dasanaike 2026, Table 3).

    Parameters
    ----------
    queries : pd.DataFrame
        The dataset to find matches for.
    corpus : pd.DataFrame
        The reference dataset to match against.
    column_query : str, optional
        Column name in queries containing the text to match.
        Mutually exclusive with columns_query.
    column_corpus : str, optional
        Column name in corpus containing the text to match.
        Defaults to column_query if not specified.
    columns_query : list of str, optional
        Multiple column names in queries to concatenate for matching.
        Mutually exclusive with column_query.
    columns_corpus : list of str, optional
        Multiple column names in corpus to concatenate for matching.
        Defaults to columns_query if not specified.
    retrieval_top_k : int
        Number of candidates to retrieve per query. Default: 20
    embedding_model : str
        Model for dense retrieval. Default: "Qwen/Qwen3-Embedding-0.6B"
    reranker_model : str
        Model for reranking. Default: "jinaai/jina-reranker-v2-base-multilingual"
    show_progress : bool
        Whether to show progress bars. Default: True
    cache_dir : str, optional
        Directory to download/cache models. Defaults to HuggingFace cache (~/.cache/huggingface).
    batch_size : int
        Number of corpus texts to embed at once. Lower values reduce peak memory
        for large corpora. Default: 50,000.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - query_idx: Index in the query DataFrame
        - query_text: The query text (concatenated if multiple columns)
        - match_idx: Index in the corpus DataFrame
        - match_text: The matched text (concatenated if multiple columns)
        - score: Confidence score

    Example
    -------
    >>> import pandas as pd
    >>> from zeroshot_linkage import link
    >>>
    >>> # Single column
    >>> queries = pd.DataFrame({"name": ["John Smith", "Jane Doe"]})
    >>> corpus = pd.DataFrame({"name": ["J. Smith", "Jane M. Doe", "Bob Wilson"]})
    >>> results = link(queries, corpus, column_query="name")
    >>>
    >>> # Multiple columns (concatenated automatically)
    >>> queries = pd.DataFrame({"city": ["OKC", "SF"], "state": ["OK", "CA"]})
    >>> corpus = pd.DataFrame({"city": ["Oklahoma City", "San Francisco"], "state": ["OK", "CA"]})
    >>> results = link(queries, corpus, columns_query=["city", "state"])
    """
    if column_query is not None and columns_query is not None:
        raise ValueError("Specify either column_query or columns_query, not both.")
    if column_query is None and columns_query is None:
        raise ValueError("Specify either column_query or columns_query.")

    # Prepare data
    queries = queries.reset_index(drop=True)
    corpus = corpus.reset_index(drop=True)

    if columns_query is not None:
        if columns_corpus is None:
            columns_corpus = columns_query
        query_texts = queries[columns_query].astype(str).agg(" | ".join, axis=1).tolist()
        corpus_texts = corpus[columns_corpus].astype(str).agg(" | ".join, axis=1).tolist()
    else:
        if column_corpus is None:
            column_corpus = column_query
        query_texts = queries[column_query].astype(str).tolist()
        corpus_texts = corpus[column_corpus].astype(str).tolist()

    # Build retrieval index
    retriever = EnsembleRetriever(
        embedding_model=embedding_model,
        top_k=retrieval_top_k,
        cache_dir=cache_dir,
    )
    retriever.index(corpus_texts, show_progress=show_progress, batch_size=batch_size)

    # Initialize reranker
    reranker = CrossEncoderReranker(model_name=reranker_model, cache_dir=cache_dir)

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


def link_blocked(
    queries: pd.DataFrame,
    corpus: pd.DataFrame,
    blocking_query: str,
    detail_query: str,
    blocking_corpus: Optional[str] = None,
    detail_corpus: Optional[str] = None,
    retrieval_top_k: int = 20,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
    batch_size: int = 50000,
) -> pd.DataFrame:
    """
    Link records using hierarchical blocking - match high-level groups first,
    then match details within those groups.

    Example: Match states first, then match counties within matched states.

    Parameters
    ----------
    queries : pd.DataFrame
        The dataset to find matches for.
    corpus : pd.DataFrame
        The reference dataset to match against.
    blocking_query : str
        Column in queries for blocking (e.g., "state").
    detail_query : str
        Column in queries for detail matching (e.g., "county").
    blocking_corpus : str, optional
        Column in corpus for blocking. Defaults to blocking_query.
    detail_corpus : str, optional
        Column in corpus for detail matching. Defaults to detail_query.
    retrieval_top_k : int
        Number of candidates to retrieve per query. Default: 20
    embedding_model : str
        Model for dense retrieval. Default: "Qwen/Qwen3-Embedding-0.6B"
    reranker_model : str
        Model for reranking. Default: "jinaai/jina-reranker-v2-base-multilingual"
    show_progress : bool
        Whether to show progress bars. Default: True
    cache_dir : str, optional
        Directory to download/cache models.
    batch_size : int
        Number of corpus texts to embed at once. Lower values reduce peak memory
        for large corpora. Default: 50,000.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - query_idx: Index in the query DataFrame
        - query_block: The query blocking value (e.g., state name)
        - query_detail: The query detail value (e.g., county name)
        - match_idx: Index in the corpus DataFrame
        - match_block: The matched blocking value
        - match_detail: The matched detail value
        - block_score: Confidence score for the block match
        - detail_score: Confidence score for the detail match

    Example
    -------
    >>> queries = pd.DataFrame({
    ...     "state": ["Kalifornia", "Texass"],
    ...     "county": ["Los Angelos", "Harris Co"]
    ... })
    >>> corpus = pd.DataFrame({
    ...     "state": ["California", "California", "Texas", "Texas"],
    ...     "county": ["Los Angeles", "San Francisco", "Harris", "Dallas"]
    ... })
    >>> results = link_blocked(
    ...     queries, corpus,
    ...     blocking_query="state", detail_query="county"
    ... )
    """
    if blocking_corpus is None:
        blocking_corpus = blocking_query
    if detail_corpus is None:
        detail_corpus = detail_query

    # Prepare data
    queries = queries.reset_index(drop=True)
    corpus = corpus.reset_index(drop=True)

    # Get corpus block values and detail texts
    corpus_blocks = corpus[blocking_corpus].astype(str).tolist()
    corpus_details = corpus[detail_corpus].astype(str).tolist()

    # Step 1: Match blocking values (e.g., states)
    unique_query_blocks = queries[blocking_query].astype(str).unique().tolist()
    unique_corpus_blocks = list(set(corpus_blocks))

    if show_progress:
        print(f"Matching {len(unique_query_blocks)} unique blocking values...")

    # Build retriever for blocking (small - just unique block names)
    block_retriever = EnsembleRetriever(
        embedding_model=embedding_model,
        top_k=min(retrieval_top_k, len(unique_corpus_blocks)),
        cache_dir=cache_dir,
    )
    block_retriever.index(unique_corpus_blocks, show_progress=show_progress)

    # Initialize reranker (shared for both stages)
    reranker = CrossEncoderReranker(model_name=reranker_model, cache_dir=cache_dir)

    # Match each unique query block to corpus blocks
    block_mapping = {}  # query_block -> (corpus_block, score)
    for query_block in unique_query_blocks:
        candidate_indices = block_retriever.retrieve(query_block)
        candidate_blocks = [unique_corpus_blocks[i] for i in candidate_indices]

        if not candidate_blocks:
            block_mapping[query_block] = (None, None)
            continue

        scores = reranker.score(query_block, candidate_blocks)
        best_idx = int(np.argmax(scores))
        block_mapping[query_block] = (candidate_blocks[best_idx], float(scores[best_idx]))

    # Step 2: Build ONE retriever for entire detail corpus (OPTIMIZED)
    if show_progress:
        print(f"Building detail index for {len(corpus_details)} records...")

    detail_retriever = EnsembleRetriever(
        embedding_model=embedding_model,
        top_k=retrieval_top_k * 5,  # Retrieve more, filter by block later
        cache_dir=cache_dir,
    )
    detail_retriever.index(corpus_details, show_progress=show_progress, batch_size=batch_size)

    # Pre-compute block membership for fast filtering
    block_to_indices = {}
    for idx, block in enumerate(corpus_blocks):
        if block not in block_to_indices:
            block_to_indices[block] = set()
        block_to_indices[block].add(idx)

    # Match each query
    if show_progress:
        print("Linking records...")

    results = []
    iterator = queries.iterrows()
    if show_progress:
        iterator = tqdm(list(iterator), desc="Linking records")

    for query_idx, row in iterator:
        query_block = str(row[blocking_query])
        query_detail = str(row[detail_query])

        matched_block, block_score = block_mapping.get(query_block, (None, None))

        if matched_block is None or matched_block not in block_to_indices:
            results.append({
                "query_idx": query_idx,
                "query_block": query_block,
                "query_detail": query_detail,
                "match_idx": None,
                "match_block": matched_block,
                "match_detail": None,
                "block_score": block_score,
                "detail_score": None,
            })
            continue

        # Retrieve candidates from full corpus, then filter by block
        all_candidate_indices = detail_retriever.retrieve(query_detail)
        valid_indices = block_to_indices[matched_block]

        # Filter to only candidates in the matched block
        candidate_indices = [i for i in all_candidate_indices if i in valid_indices]

        if not candidate_indices:
            # Fallback: if no candidates after filtering, try all in block
            candidate_indices = list(valid_indices)[:retrieval_top_k]

        candidate_texts = [corpus_details[i] for i in candidate_indices]

        if not candidate_texts:
            results.append({
                "query_idx": query_idx,
                "query_block": query_block,
                "query_detail": query_detail,
                "match_idx": None,
                "match_block": matched_block,
                "match_detail": None,
                "block_score": block_score,
                "detail_score": None,
            })
            continue

        # Rerank detail candidates
        detail_scores = reranker.score(query_detail, candidate_texts)
        detail_scores = np.atleast_1d(detail_scores)  # Handle scalar case
        best_local_idx = int(np.argmax(detail_scores))
        best_detail_score = float(detail_scores[best_local_idx])
        match_idx = candidate_indices[best_local_idx]
        match_detail = candidate_texts[best_local_idx]

        results.append({
            "query_idx": query_idx,
            "query_block": query_block,
            "query_detail": query_detail,
            "match_idx": match_idx,
            "match_block": matched_block,
            "match_detail": match_detail,
            "block_score": block_score,
            "detail_score": best_detail_score,
        })

    return pd.DataFrame(results)
