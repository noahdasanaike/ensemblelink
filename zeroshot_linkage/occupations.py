"""
Zero-shot occupational classification using EnsembleLink components.
"""

from typing import Callable, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .retrieval import EnsembleRetriever
from .reranker import CrossEncoderReranker


def _as_text(title: str, description: Optional[str]) -> str:
    title = "" if title is None else str(title)
    description = "" if description is None else str(description)
    description = description.strip()
    if description:
        return f"{title} :: {description}"
    return title


def build_unified_corpus(
    hisco: pd.DataFrame,
    isco: pd.DataFrame,
    hisco_title_col: str = "title",
    hisco_desc_col: str = "description",
    hisco_code_col: str = "hisco_code",
    isco_title_col: str = "title",
    isco_desc_col: str = "description",
    isco_code_col: str = "isco_code",
    era_labels: Tuple[str, str] = ("historical", "modern"),
) -> pd.DataFrame:
    """
    Build a unified corpus spanning historical (HISCO) and modern (ISCO) titles.

    Returns a DataFrame with columns:
        - title
        - description
        - era
        - canonical_code
        - hisco_code
        - isco_code
        - text (title + description)
    """
    hisco_df = hisco.copy()
    hisco_df["title"] = hisco_df[hisco_title_col].astype(str)
    hisco_df["description"] = (
        hisco_df[hisco_desc_col].astype(str) if hisco_desc_col in hisco_df else ""
    )
    hisco_df["era"] = era_labels[0]
    hisco_df["hisco_code"] = hisco_df[hisco_code_col] if hisco_code_col in hisco_df else None
    hisco_df["isco_code"] = None
    hisco_df["canonical_code"] = hisco_df["hisco_code"]

    isco_df = isco.copy()
    isco_df["title"] = isco_df[isco_title_col].astype(str)
    isco_df["description"] = (
        isco_df[isco_desc_col].astype(str) if isco_desc_col in isco_df else ""
    )
    isco_df["era"] = era_labels[1]
    isco_df["isco_code"] = isco_df[isco_code_col] if isco_code_col in isco_df else None
    isco_df["hisco_code"] = None
    isco_df["canonical_code"] = isco_df["isco_code"]

    unified = pd.concat([hisco_df, isco_df], ignore_index=True)
    unified["text"] = [
        _as_text(t, d) for t, d in zip(unified["title"], unified["description"])
    ]

    cols = [
        "title",
        "description",
        "era",
        "canonical_code",
        "hisco_code",
        "isco_code",
        "text",
    ]
    return unified[cols]


class EnsembleOcc:
    """
    Universal occupational classifier using EnsembleLink retrieval + reranking.
    """

    def __init__(
        self,
        embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
        reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
        retrieval_top_k: int = 30,
        ngram_range: tuple = (2, 4),
        max_features: int = 50000,
        cache_dir: Optional[str] = None,
        device: Optional[str] = None,
    ):
        self.embedding_model = embedding_model
        self.reranker_model = reranker_model
        self.retrieval_top_k = retrieval_top_k
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.cache_dir = cache_dir
        self.device = device
        self._retriever: Optional[EnsembleRetriever] = None
        self._reranker: Optional[CrossEncoderReranker] = None
        self._corpus: Optional[pd.DataFrame] = None
        self._corpus_texts: List[str] = []

    def index(self, corpus: pd.DataFrame, text_col: str = "text", show_progress: bool = True) -> None:
        self._corpus = corpus.reset_index(drop=True)
        self._corpus_texts = self._corpus[text_col].astype(str).tolist()

        self._retriever = EnsembleRetriever(
            embedding_model=self.embedding_model,
            top_k=self.retrieval_top_k,
            ngram_range=self.ngram_range,
            max_features=self.max_features,
            device=self.device,
            cache_dir=self.cache_dir,
        )
        self._retriever.index(self._corpus_texts, show_progress=show_progress)

        self._reranker = CrossEncoderReranker(
            model_name=self.reranker_model,
            device=self.device,
            cache_dir=self.cache_dir,
        )

    def predict(
        self,
        queries: Iterable[str],
        second_stage_reranker: Optional[Callable[[str, Sequence[str]], int]] = None,
        second_stage_top_n: int = 5,
        show_progress: bool = True,
    ) -> pd.DataFrame:
        """
        Predict the best occupational match for each query.

        second_stage_reranker:
            Optional callable that selects the best index among a shortlist
            (e.g., an LLM reranker). Signature: (query, candidate_texts) -> int.
        """
        if self._retriever is None or self._reranker is None or self._corpus is None:
            raise ValueError("Call index() before predict().")

        results = []
        query_list = [str(q) for q in queries]

        iterator = enumerate(query_list)
        if show_progress:
            from tqdm import tqdm

            iterator = tqdm(list(iterator), desc="Classifying occupations")

        for query_idx, query in iterator:
            candidate_indices = self._retriever.retrieve(query)
            candidate_texts = [self._corpus_texts[i] for i in candidate_indices]

            if not candidate_texts:
                results.append({
                    "query_idx": query_idx,
                    "query_text": query,
                    "match_idx": None,
                    "match_text": None,
                    "score": None,
                })
                continue

            scores = self._reranker.score(query, candidate_texts)
            scores = np.atleast_1d(scores)

            best_local_idx = int(np.argmax(scores))

            if second_stage_reranker is not None:
                # Optional LLM reranker over top-N candidates
                top_n = int(min(second_stage_top_n, len(candidate_texts)))
                top_indices = np.argsort(-scores)[:top_n]
                top_texts = [candidate_texts[i] for i in top_indices]
                chosen = int(second_stage_reranker(query, top_texts))
                if 0 <= chosen < len(top_texts):
                    best_local_idx = top_indices[chosen]

            best_score = float(scores[best_local_idx])
            match_idx = candidate_indices[best_local_idx]
            match_text = candidate_texts[best_local_idx]

            results.append({
                "query_idx": query_idx,
                "query_text": query,
                "match_idx": match_idx,
                "match_text": match_text,
                "score": best_score,
            })

        return pd.DataFrame(results)


class _UnionFind:
    def __init__(self, n: int):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            self.parent[ra] = rb
        elif self.rank[ra] > self.rank[rb]:
            self.parent[rb] = ra
        else:
            self.parent[rb] = ra
            self.rank[ra] += 1


def deduplicate_corpus(
    corpus: pd.DataFrame,
    text_col: str = "text",
    era_col: str = "era",
    code_cols: Sequence[str] = ("hisco_code", "isco_code", "canonical_code"),
    score_threshold: float = 0.95,
    retrieval_top_k: int = 30,
    embedding_model: str = "Qwen/Qwen3-Embedding-0.6B",
    reranker_model: str = "jinaai/jina-reranker-v2-base-multilingual",
    show_progress: bool = True,
    cache_dir: Optional[str] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Deduplicate a corpus by linking it to itself and merging high-score pairs.

    Returns:
        deduped_df: canonicalized corpus
        mapping_df: original index -> canonical index
    """
    corpus = corpus.reset_index(drop=True)
    texts = corpus[text_col].astype(str).tolist()

    retriever = EnsembleRetriever(
        embedding_model=embedding_model,
        top_k=retrieval_top_k,
        cache_dir=cache_dir,
    )
    retriever.index(texts, show_progress=show_progress)

    reranker = CrossEncoderReranker(
        model_name=reranker_model,
        cache_dir=cache_dir,
    )

    uf = _UnionFind(len(corpus))

    iterator = range(len(corpus))
    if show_progress:
        from tqdm import tqdm

        iterator = tqdm(list(iterator), desc="Deduplicating corpus")

    for i in iterator:
        query = texts[i]
        candidate_indices = retriever.retrieve(query)
        candidate_indices = [j for j in candidate_indices if j != i and j > i]

        if not candidate_indices:
            continue

        candidate_texts = [texts[j] for j in candidate_indices]
        scores = reranker.score(query, candidate_texts)
        scores = np.atleast_1d(scores)

        for j, score in zip(candidate_indices, scores):
            if float(score) >= score_threshold:
                uf.union(i, j)

    # Build clusters
    clusters = {}
    for idx in range(len(corpus)):
        root = uf.find(idx)
        clusters.setdefault(root, []).append(idx)

    # Choose canonical record per cluster
    canonical_indices = {}
    for root, members in clusters.items():
        canonical = members[0]
        if era_col in corpus.columns:
            moderns = [m for m in members if str(corpus.loc[m, era_col]) == "modern"]
            if moderns:
                canonical = moderns[0]
        canonical_indices[root] = canonical

    mapping_rows = []
    for root, members in clusters.items():
        canonical = canonical_indices[root]
        for idx in members:
            mapping_rows.append({
                "orig_idx": idx,
                "canonical_idx": canonical,
                "cluster_id": root,
            })

    mapping_df = pd.DataFrame(mapping_rows)

    # Aggregate clusters
    dedup_rows = []
    for root, members in clusters.items():
        canonical = canonical_indices[root]
        row = corpus.loc[canonical].to_dict()

        aliases = [str(corpus.loc[m, "title"]) for m in members if "title" in corpus.columns]
        if aliases:
            row["aliases"] = " | ".join(sorted(set(aliases)))
        row["n_records"] = len(members)

        for col in code_cols:
            if col in corpus.columns:
                values = [str(corpus.loc[m, col]) for m in members if pd.notna(corpus.loc[m, col])]
                values = [v for v in values if v not in ("", "None", "nan")]
                row[col] = " | ".join(sorted(set(values))) if values else None

        dedup_rows.append(row)

    deduped_df = pd.DataFrame(dedup_rows).reset_index(drop=True)

    return deduped_df, mapping_df
