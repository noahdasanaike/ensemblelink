"""
Cross-encoder reranking for candidate scoring.

A cross-encoder reads the query and a candidate together and scores how well
they match. Cross-encoders are more accurate than bi-encoders but slower, so
they run only over the small retrieved pool. EnsembleLink uses two
complementary rerankers (Jina v2 and BGE v2-m3) whose raw scores become one of
the four fusion experts; :class:`RerankerEnsemble` exposes their per-model raw
scores so the fusion stage can z-normalize and sum them. :class:`CrossEncoderReranker`
remains for lighter, single-model use.
"""

import numpy as np
from typing import List, Optional


def _is_jina(name: str) -> bool:
    return "jina" in name.lower()


class CrossEncoderReranker:
    """
    Single cross-encoder reranker.

    Parameters
    ----------
    model_name : str
        HuggingFace model name for the cross-encoder.
    max_length : int
        Maximum tokenized pair length. Short names (people, places, orgs) match
        well at 128; raise it for longer fields such as bibliographic records.
    device : str, optional
        Device for inference ("cuda" or "cpu").
    cache_dir : str, optional
        Directory to download/cache models. Defaults to the HuggingFace cache.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        max_length: int = 512,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self._model = None
        self._kind = "jina" if _is_jina(model_name) else "cross-encoder"

    def _load_model(self):
        """Lazy-load the reranker model."""
        if self._model is not None:
            return
        import torch

        if self.device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"

        if self._kind == "jina":
            # Jina v2 exposes a bespoke compute_score() via its custom code.
            from transformers import AutoModelForSequenceClassification

            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            self._model.to(self.device)
            self._model.eval()
        else:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                self.model_name,
                max_length=self.max_length,
                trust_remote_code=True,
                cache_folder=self.cache_dir,
                device=self.device,
            )

    def score(self, query: str, candidates: List[str]) -> np.ndarray:
        """Score a single query against its candidate list."""
        if not candidates:
            return np.array([])
        pairs = [[query, c] for c in candidates]
        return self.score_pairs(pairs)

    def score_pairs(self, pairs, batch_size: int = 256, show_progress: bool = False) -> np.ndarray:
        """Score a flat list of [query, candidate] pairs."""
        if not pairs:
            return np.array([])
        self._load_model()
        if self._kind == "jina":
            scores = self._model.compute_score(pairs, max_length=self.max_length)
        else:
            scores = self._model.predict(
                pairs, batch_size=batch_size, show_progress_bar=show_progress
            )
        # A single pair can come back as a Python float / 0-d array; force 1-d.
        return np.atleast_1d(np.asarray(scores, dtype=np.float64))


class RerankerEnsemble:
    """
    Ensemble of cross-encoder rerankers.

    Loads every named reranker and returns each one's raw scores separately, so
    the fusion stage can z-normalize per query pool and sum (a z-score sum is
    scale-free, so the two rerankers contribute equally regardless of their
    native score ranges).

    Parameters
    ----------
    model_names : list of str
        Cross-encoder model names. Default is Jina v2 multilingual and BGE v2-m3.
    max_length : int
        Maximum tokenized pair length, shared across rerankers.
    device : str, optional
        Device for inference ("cuda" or "cpu").
    cache_dir : str, optional
        Directory to download/cache models. Defaults to the HuggingFace cache.
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        max_length: int = 128,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        if model_names is None:
            model_names = [
                "jinaai/jina-reranker-v2-base-multilingual",
                "BAAI/bge-reranker-v2-m3",
            ]
        self.model_names = list(model_names)
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self._rerankers: Optional[List[CrossEncoderReranker]] = None

    def _load(self):
        if self._rerankers is None:
            self._rerankers = [
                CrossEncoderReranker(
                    model_name=name,
                    max_length=self.max_length,
                    device=self.device,
                    cache_dir=self.cache_dir,
                )
                for name in self.model_names
            ]

    def score_pairs(self, pairs, batch_size: int = 256, show_progress: bool = False) -> List[np.ndarray]:
        """Score [query, candidate] pairs with every reranker.

        Returns one raw score array per reranker (same order as ``model_names``).
        """
        self._load()
        if not pairs:
            return [np.array([]) for _ in self._rerankers]
        return [
            r.score_pairs(pairs, batch_size=batch_size, show_progress=show_progress)
            for r in self._rerankers
        ]
