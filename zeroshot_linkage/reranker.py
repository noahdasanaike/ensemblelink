"""
Cross-encoder reranking for candidate scoring.
"""

import numpy as np
from typing import List, Optional


class CrossEncoderReranker:
    """
    Cross-encoder reranker for scoring query-candidate pairs.

    Uses a cross-encoder model to score how well each candidate
    matches the query. Cross-encoders are more accurate than
    bi-encoders but slower, so they're used after retrieval
    reduces the candidate set.

    Parameters
    ----------
    model_name : str
        HuggingFace model name for the cross-encoder.
    device : str, optional
        Device for inference ("cuda" or "cpu").
    cache_dir : str, optional
        Directory to download/cache models. Defaults to HuggingFace cache.
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v2-base-multilingual",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self.cache_dir = cache_dir
        self._model = None

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            import torch
            from transformers import AutoModelForSequenceClassification

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Jina v2 Multilingual uses AutoModelForSequenceClassification
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype="auto",
                trust_remote_code=True,
                cache_dir=self.cache_dir,
            )
            self._model.to(self.device)
            self._model.eval()

    def score(self, query: str, candidates: List[str]) -> np.ndarray:
        """
        Score candidates for a query.

        Parameters
        ----------
        query : str
            The query text.
        candidates : List[str]
            List of candidate texts to score.

        Returns
        -------
        np.ndarray
            Array of scores, one per candidate. Higher is better.
        """
        if not candidates:
            return np.array([])

        self._load_model()

        # Use Jina v2's compute_score method
        pairs = [[query, c] for c in candidates]
        scores = self._model.compute_score(pairs, max_length=1024)

        return np.array(scores)
