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
    """

    def __init__(
        self,
        model_name: str = "jinaai/jina-reranker-v3",
        device: Optional[str] = None,
    ):
        self.model_name = model_name
        self.device = device
        self._model = None

    def _load_model(self):
        """Lazy load the reranker model."""
        if self._model is None:
            import torch
            from transformers import AutoModel

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            # Jina v3 requires AutoModel with dtype="auto" and trust_remote_code
            self._model = AutoModel.from_pretrained(
                self.model_name,
                dtype="auto",
                trust_remote_code=True,
                device_map=self.device,
            )
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

        # Use Jina v3's built-in rerank method
        results = self._model.rerank(query, candidates)

        # Build score array matching original candidate order
        # Results come back sorted by relevance, but we need scores in original order
        doc_to_score = {r['document']: r['relevance_score'] for r in results}
        scores = np.array([doc_to_score.get(c, 0.0) for c in candidates])

        return scores
