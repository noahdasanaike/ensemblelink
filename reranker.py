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
            from transformers import AutoModelForSequenceClassification, AutoTokenizer

            if self.device is None:
                self.device = "cuda" if torch.cuda.is_available() else "cpu"

            self._tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            # Set pad token if not defined (required for batch processing)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            self._model = AutoModelForSequenceClassification.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                trust_remote_code=True,
            )
            # Set model's pad token id
            if self._model.config.pad_token_id is None:
                self._model.config.pad_token_id = self._tokenizer.pad_token_id
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

        import torch

        # Create pairs
        pairs = [[query, cand] for cand in candidates]

        # Tokenize
        inputs = self._tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=512,
            return_tensors="pt",
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Score
        with torch.no_grad():
            outputs = self._model(**inputs)
            # Handle different output formats
            if hasattr(outputs, "logits"):
                scores = outputs.logits
            else:
                scores = outputs[0]

            # Convert to probabilities if classification head
            if scores.shape[-1] == 1:
                scores = scores.squeeze(-1)
            else:
                # Binary classification: take positive class probability
                scores = torch.softmax(scores, dim=-1)[:, 1]

            scores = scores.cpu().numpy()

        return scores
