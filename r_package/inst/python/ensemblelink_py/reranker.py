"""
Cross-encoder rerankers.

A cross-encoder reads the query and a candidate together and scores how well
they match. EnsembleLink's reranker expert is built from Jina Reranker v2
(multilingual) and BGE Reranker v2-m3 by default; each pair score is the
sigmoid of the model's relevance logit (a 0-1 probability), computed as the
paper's benchmark computes it:

* ``AutoModelForSequenceClassification`` in bfloat16 on a GPU (float32 on CPU);
  Jina v2 with its own code and ``use_flash_attn=False``, other models with
  PyTorch SDPA attention;
* pairs tokenized as (query, candidate), padded to the longest pair of each
  batch, in batches of ``batch_size`` (default 128) in the order given;
* score = ``sigmoid(logit)`` computed in float32 and stored as float32.

zerank-2 (``zeroentropy/zerank-2-reranker``, Apache-2.0, 4B parameters) is an
optional third reranker: a Sentence Transformers ``CrossEncoder`` in bfloat16
whose raw "Yes" logit is mapped to the card's documented 0-1 score
``sigmoid(logit / 5)``. It is not in the default because it is about nine times
slower per pair than Jina v2 and BGE together and did not pass the paper's
development rule (see README).

Default models are loaded at the Hugging Face revisions used in the paper.
"""

from typing import List, Optional

import numpy as np

JINA_RERANKER = "jinaai/jina-reranker-v2-base-multilingual"
BGE_RERANKER = "BAAI/bge-reranker-v2-m3"
ZERANK_RERANKER = "zeroentropy/zerank-2-reranker"
# Hugging Face commits used in the paper's benchmark (configs/ensemblelink_development_v1.json).
MODEL_REVISIONS = {
    "microsoft/harrier-oss-v1-0.6b": "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee",
    JINA_RERANKER: "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9",
    BGE_RERANKER: "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    ZERANK_RERANKER: "5eae30d5ee3c6b2df2ef6d723bde45172d761c4c",
}
ZERANK_TEMPERATURE = 5.0
ZERANK_BATCH = 64
ZERANK_CHUNK = 20000  # pairs per predict call, as in the benchmark
PAIR_CHUNK = 4096     # the benchmark scores pairs in chunks of 4,096, batched within each chunk
DEFAULT_RERANKER_BATCH = 128
DEFAULT_RERANKER_MAX_LENGTH = 1024


def _is_jina(name: str) -> bool:
    return "jina" in name.lower()


def _is_zerank(name: str) -> bool:
    return "zerank" in name.lower()


def model_revision(name: str, revision: Optional[str] = None) -> Optional[str]:
    """Pinned revision of a default model (None for other models, i.e. the latest)."""
    return revision if revision is not None else MODEL_REVISIONS.get(name)


def resolve_device(device: Optional[str]) -> str:
    import torch

    if device in (None, "auto"):
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def resolve_dtype(dtype, device: str):
    """"auto": bfloat16 on a GPU (the paper's precision), float32 on CPU."""
    import torch

    if dtype in (None, "auto"):
        return torch.bfloat16 if str(device).startswith("cuda") else torch.float32
    if isinstance(dtype, str):
        return {"bfloat16": torch.bfloat16, "bf16": torch.bfloat16, "float16": torch.float16,
                "fp16": torch.float16, "float32": torch.float32, "fp32": torch.float32}[dtype]
    return dtype


def zerank_probability(logits) -> np.ndarray:
    """zerank-2's documented 0-1 relevance score from its raw logit (float32)."""
    logits = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-logits / ZERANK_TEMPERATURE))).astype(np.float32)


class CrossEncoderReranker:
    """
    One cross-encoder reranker returning 0-1 pair probabilities.

    Parameters
    ----------
    model_name : str
        Hugging Face model id: a sequence-classification cross-encoder with one
        relevance logit (Jina v2, BGE v2-m3 and similar), or zerank-2.
    max_length : int
        Token ceiling per (query, candidate) pair; longer pairs are truncated.
        Pairs below it are unaffected (padding is to the longest pair of a batch).
    device : str, optional
        "cuda", "cpu" or None/"auto".
    cache_dir : str, optional
        Model cache directory (default: the Hugging Face cache).
    batch_size : int
        Pairs per forward pass (default 128).
    dtype : str or torch.dtype
        "auto" (default): bfloat16 on GPU, float32 on CPU.
    revision : str, optional
        Model revision; defaults to the pinned revision for the default models.
    """

    def __init__(
        self,
        model_name: str = JINA_RERANKER,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_RERANKER_BATCH,
        dtype="auto",
        revision: Optional[str] = None,
    ):
        self.model_name = model_name
        self.max_length = int(max_length)
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = int(batch_size)
        self.dtype = dtype
        self.revision = model_revision(model_name, revision)
        self._model = None
        self._tokenizer = None
        self._kind = "jina" if _is_jina(model_name) else ("zerank" if _is_zerank(model_name) else "sequence-classification")

    def _load_model(self):
        if self._model is not None:
            return
        import torch

        self.device = resolve_device(self.device)
        torch_dtype = resolve_dtype(self.dtype, self.device)
        if self._kind == "zerank":
            import transformers
            from sentence_transformers import CrossEncoder

            key = "dtype" if int(transformers.__version__.split(".")[0]) >= 5 else "torch_dtype"
            self._model = CrossEncoder(self.model_name, revision=self.revision, cache_folder=self.cache_dir,
                                       device=self.device, model_kwargs={key: torch_dtype})
            return
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        kwargs = {"use_flash_attn": False} if self._kind == "jina" else {"attn_implementation": "sdpa"}
        common = dict(revision=self.revision, cache_dir=self.cache_dir, trust_remote_code=self._kind == "jina")
        try:
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=torch_dtype,
                                                                       **common, **kwargs)
        except (TypeError, ValueError):
            if self._kind == "jina":
                raise
            # A model without SDPA support: default attention.
            model = AutoModelForSequenceClassification.from_pretrained(self.model_name, torch_dtype=torch_dtype, **common)
        self._model = model.to(self.device).eval()
        self._tokenizer = AutoTokenizer.from_pretrained(self.model_name, **common)

    def _forward(self, queries: List[str], candidates: List[str]) -> np.ndarray:
        import torch

        inputs = self._tokenizer(queries, candidates, padding=True, truncation=True, max_length=self.max_length,
                                 return_tensors="pt").to(self.device)
        with torch.inference_mode():
            return self._model(**inputs).logits.float().view(-1).sigmoid().cpu().numpy()

    def score(self, query: str, candidates: List[str]) -> np.ndarray:
        """Score one query against its candidates."""
        if not candidates:
            return np.array([], dtype=np.float32)
        return self.score_pairs([[query, c] for c in candidates])

    def score_pairs(self, pairs, show_progress: bool = False, sort_by_length: bool = False) -> np.ndarray:
        """0-1 scores (float32) of [query, candidate] pairs, in input order.

        ``sort_by_length=False`` (exact mode): batches of consecutive pairs, as in the
        paper's benchmark. ``True`` (fast mode): pairs are scored longest first, so each
        batch pads to a similar length; bfloat16 scores then move at noise level.
        """
        n = len(pairs)
        if n == 0:
            return np.array([], dtype=np.float32)
        self._load_model()
        if self._kind == "zerank":
            out = np.empty(n, dtype=np.float32)
            for start in range(0, n, ZERANK_CHUNK):
                chunk = [tuple(p) for p in pairs[start:start + ZERANK_CHUNK]]
                raw = self._model.predict(chunk, batch_size=ZERANK_BATCH, convert_to_numpy=True, show_progress_bar=False)
                out[start:start + len(chunk)] = zerank_probability(np.asarray(raw, dtype=np.float64).reshape(-1))
            return out
        order = None
        if sort_by_length and n > 1:
            order = np.argsort(-np.fromiter((len(p[0]) + len(p[1]) for p in pairs), dtype=np.int64, count=n), kind="stable")
            pairs = [pairs[i] for i in order]
        out = np.empty(n, dtype=np.float32)
        pbar = None
        if show_progress:
            from tqdm import tqdm

            pbar = tqdm(total=n, desc="Reranking (" + self.model_name.split("/")[-1] + ")")
        b = self.batch_size
        for c0 in range(0, n, PAIR_CHUNK):
            chunk = pairs[c0:c0 + PAIR_CHUNK]
            for j in range(0, len(chunk), b):
                part = chunk[j:j + b]
                out[c0 + j:c0 + j + len(part)] = self._forward([p[0] for p in part], [p[1] for p in part])
                if pbar is not None:
                    pbar.update(len(part))
        if pbar is not None:
            pbar.close()
        if order is not None:
            restored = np.empty_like(out)
            restored[order] = out
            out = restored
        return out


class RerankerEnsemble:
    """
    The rerankers forming the reranker expert; returns each model's 0-1 scores separately,
    so fusion can z-score them per pool and sum them.

    Parameters
    ----------
    model_names : list of str
        Default: Jina v2 and BGE v2-m3.
    max_length, device, cache_dir, batch_size, dtype
        As in :class:`CrossEncoderReranker`.
    """

    def __init__(
        self,
        model_names: Optional[List[str]] = None,
        max_length: int = DEFAULT_RERANKER_MAX_LENGTH,
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        batch_size: int = DEFAULT_RERANKER_BATCH,
        dtype="auto",
    ):
        self.model_names = list(model_names) if model_names is not None else [JINA_RERANKER, BGE_RERANKER]
        if not self.model_names:
            raise ValueError("At least one reranker is required")
        self.max_length = max_length
        self.device = device
        self.cache_dir = cache_dir
        self.batch_size = batch_size
        self.dtype = dtype
        self._rerankers: Optional[List[CrossEncoderReranker]] = None

    def _load(self):
        if self._rerankers is None:
            self._rerankers = [
                CrossEncoderReranker(model_name=name, max_length=self.max_length, device=self.device,
                                     cache_dir=self.cache_dir, batch_size=self.batch_size, dtype=self.dtype)
                for name in self.model_names
            ]

    def score_pairs(self, pairs, show_progress: bool = False, sort_by_length: bool = False) -> List[np.ndarray]:
        """One float32 0-1 score array per reranker (order of ``model_names``)."""
        self._load()
        return [r.score_pairs(pairs, show_progress=show_progress, sort_by_length=sort_by_length)
                for r in self._rerankers]
