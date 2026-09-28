"""Batching modes, deduplication, exact TF-IDF and the index cache, with fake models.

The models are replaced by deterministic, batch-invariant fakes whose embeddings
have four entries of +-1/2, so every dot product is exact in float32. With such
models exact=True and exact=False must give identical results.

  python -m pytest tests -q
"""
import hashlib
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from zeroshot_linkage import core, retrieval, reranker, index_cache  # noqa: E402
from zeroshot_linkage import _fast_tfidf  # noqa: E402
from zeroshot_linkage.fusion import lexical_text  # noqa: E402
from zeroshot_linkage.retrieval import unique_inverse, EnsembleRetriever  # noqa: E402

DIM = 8


# --------------------------------------------------------------------------- fakes

class FakeEmbedder:
    """Deterministic per-text embeddings, independent of batch composition; records every call."""

    def __init__(self):
        self.seen = []
        self.calls = []
        self.batch_sizes = []

    @staticmethod
    def vec(text, prompt=None):
        # Four entries of +-1/2: unit norm and dot products exact in float32, so no batching can move them.
        h = hashlib.sha256(((prompt or "") + text).encode()).digest()
        pos = np.argsort(np.frombuffer(h[:DIM], dtype=np.uint8), kind="stable")[:4]
        v = np.zeros(DIM, dtype=np.float32)
        v[pos] = np.where(np.frombuffer(h[DIM:DIM + 4], dtype=np.uint8) % 2 == 0, 0.5, -0.5)
        return v

    def encode(self, texts, prompt=None, batch_size=32, normalize_embeddings=True,
               show_progress_bar=False, convert_to_numpy=True, **kw):
        texts = list(texts)
        self.seen.extend(texts)
        self.calls.append(texts)
        self.batch_sizes.append(batch_size)
        if not texts:
            return np.empty((0, DIM), dtype=np.float32)
        return np.stack([self.vec(t, prompt) for t in texts])

    def get_embedding_dimension(self):
        return DIM


def fake_pair_score(q, c, salt):
    from rapidfuzz.distance import JaroWinkler
    return np.float32(round(JaroWinkler.similarity(q[::-1], c[::-1]) * 64 + salt * len(c) % 7) / 448.0)


def install_fakes(matcher):
    """Swap the models of a FusionMatcher for the fakes; returns (embedder, reranker call log)."""
    emb = FakeEmbedder()
    matcher.retriever._embed_model = emb
    log = {"pairs": [], "sorted": []}

    def score_pairs(pairs, show_progress=False, sort_by_length=False, **kw):
        log["pairs"].append([tuple(p) for p in pairs])
        log["sorted"].append(sort_by_length)
        return [np.array([fake_pair_score(q, c, s) for q, c in pairs], dtype=np.float32) for s in (1, 2)]

    matcher.reranker.score_pairs = score_pairs
    return emb, log


def dataset(seed=0):
    rng = np.random.default_rng(seed)
    first = ["ANNA", "JOHN", "MARIA", "JOSE", "LI", "WEI", "OLGA", "IVAN", "SARA", "OMAR"]
    last = ["SMITH", "GARCIA", "IVANOV", "WANG", "NGUYEN", "KOWALSKI", "OKAFOR", "SILVA"]
    corpus = [f"{rng.choice(first)} {rng.choice(last)} {rng.integers(1940, 2005)}" for _ in range(400)]
    corpus += corpus[:60]            # exact duplicate records
    corpus += ["Москва", "Moskva", "東京", "Tokyo"]
    rng.shuffle(corpus)
    base = [c.replace("A", "E", 1) for c in corpus[:40]] + ["Moscow", "Москва", "@@@", "東京都"]
    queries = [base[i] for i in rng.integers(0, len(base), 150)]   # heavy duplication
    return queries, corpus


def run_matcher(exact=True, index_cache_dir=None, queries=None, corpus=None, prompt=None, **kw):
    if index_cache_dir is not None:
        kw["index_cache"] = str(index_cache_dir)
    m = core.FusionMatcher(pool_size=5, exact=exact, device="cpu", **kw)
    emb, log = install_fakes(m)
    res = m.link(queries, corpus, show_progress=False, query_prompt=prompt)
    return res, m, emb, log


def same(a, b):
    for k in ["match_idx", "score", "margin", "reranker_probability", "fused_score", "top_tie_count"]:
        np.testing.assert_array_equal(a[k], b[k], err_msg=k)


# --------------------------------------------------------------------------- dedupe helpers

def test_unique_inverse_order_and_mapping():
    items = ["b", "a", "b", "c", "a", "b"]
    uniq, inv = unique_inverse(items)
    assert uniq == ["b", "a", "c"]
    assert [uniq[i] for i in inv] == items
    uniq, inv = unique_inverse(["x", "y", "z"])
    assert uniq == ["x", "y", "z"] and inv.tolist() == [0, 1, 2]


def test_encode_fast_embeds_each_distinct_text_once_and_restores_order():
    r = EnsembleRetriever(exact=False, device="cpu")
    emb = FakeEmbedder()
    r._embed_model = emb
    texts = ["short", "a much longer text " * 20, "short", "mid length", "x", "mid length", "a much longer text " * 20]
    out = r.encode(texts, show_progress=False, batch_size=2, prompt="P: ")
    assert sorted(emb.seen) == sorted(set(texts))                        # once each
    for i, t in enumerate(texts):                                         # original order, unit norm
        v = FakeEmbedder.vec(t, "P: ")
        np.testing.assert_array_equal(out[i], v)
    assert all(bs >= retrieval.EMBED_MIN_BATCH for bs in emb.batch_sizes)


def test_encode_exact_is_the_benchmark_path(monkeypatch):
    monkeypatch.setattr(retrieval, "EXACT_EMBED_CHUNK", 2)
    r = EnsembleRetriever(exact=True, device="cpu")
    emb = FakeEmbedder()
    r._embed_model = emb
    texts = ["a", "b", "a", "c", "d"]
    out = r.encode(texts, show_progress=False)
    # Calls of EXACT_EMBED_CHUNK texts in input order, duplicates included, at batch size 256.
    assert emb.calls == [["a", "b"], ["a", "c"], ["d"]]
    assert set(emb.batch_sizes) == {retrieval.DEFAULT_EMBED_BATCH}
    assert out.dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(out, axis=1), 1, rtol=1e-6)


def test_reranker_sorted_scores_come_back_in_input_order():
    rr = reranker.CrossEncoderReranker(reranker.BGE_RERANKER, batch_size=2, device="cpu")
    batches = []

    def forward(qs, cs):
        batches.append(list(zip(qs, cs)))
        return np.array([len(q) * 1000 + len(c) for q, c in zip(qs, cs)], dtype=np.float32)

    rr._model = object()
    rr._forward = forward
    pairs = [["aa", "b"], ["a", "bbbbbb"], ["aaaa", "bb"], ["a", "b"], ["abc", "abc"]]
    got = rr.score_pairs(pairs, sort_by_length=True)
    assert got.tolist() == [len(q) * 1000 + len(c) for q, c in pairs]
    flat = [p for b in batches for p in b]
    assert [len(q) + len(c) for q, c in flat] == sorted([len(q) + len(c) for q, c in pairs], reverse=True)
    batches.clear()
    got = rr.score_pairs(pairs)                     # exact: consecutive batches of batch_size, input order
    assert [len(b) for b in batches] == [2, 2, 1] and [p for b in batches for p in b] == [tuple(p) for p in pairs]


# --------------------------------------------------------------------------- exact TF-IDF

@pytest.mark.parametrize("max_features", [None, 500])
def test_parallel_tfidf_is_bit_identical(monkeypatch, max_features):
    from sklearn.feature_extraction.text import TfidfVectorizer
    monkeypatch.setattr(_fast_tfidf, "PARALLEL_MIN_DOCS", 10)
    queries, corpus = dataset(1)
    docs = [lexical_text(t) for t in corpus * 3]
    a = TfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True, max_features=max_features, dtype=np.float32)
    b = _fast_tfidf.ParallelTfidfVectorizer(analyzer="char", ngram_range=(2, 4), lowercase=True,
                                            max_features=max_features, n_jobs=2, dtype=np.float32)
    A, B = a.fit_transform(docs), b.fit_transform(docs)
    assert a.vocabulary_ == b.vocabulary_
    assert np.array_equal(a.idf_, b.idf_)
    for x, y in [(A.indptr, B.indptr), (A.indices, B.indices), (A.data, B.data)]:
        assert x.dtype == y.dtype and np.array_equal(x, y)
    q = [lexical_text(t) for t in queries]
    assert (a.transform(q) != b.transform(q)).nnz == 0


# --------------------------------------------------------------------------- matcher equivalence

@pytest.mark.parametrize("prompt", [None, "Instruct: x\nQuery: "])
def test_exact_and_fast_agree_with_batch_invariant_models(prompt):
    queries, corpus = dataset(2)
    exact, _, emb_e, log_e = run_matcher(exact=True, queries=queries, corpus=corpus, prompt=prompt)
    fast, m_f, emb_f, log_f = run_matcher(exact=False, queries=queries, corpus=corpus, prompt=prompt)
    same(exact, fast)
    # fast mode: each distinct text embedded once per role, each distinct pair scored once, length-sorted
    assert sorted(emb_f.seen) == sorted(list(set(corpus)) + list(set(queries)))
    pairs = log_f["pairs"][0]
    assert len(pairs) == len(set(pairs)) < len(log_e["pairs"][0])
    assert log_f["sorted"] == [True] and log_e["sorted"] == [False]


def test_duplicate_queries_can_break_ties_differently_but_share_scores():
    queries, corpus = dataset(3)
    res, *_ = run_matcher(exact=True, queries=queries, corpus=corpus)
    by_text = {}
    for q, f, m in zip(queries, res["fused_score"], res["margin"]):
        prev = by_text.setdefault(q, (f, m))
        assert prev[0] == f and (prev[1] == m or (np.isnan(prev[1]) and np.isnan(m)))


def test_pools_are_sorted_union_and_dense_refill():
    queries, corpus = dataset(4)
    res, m, *_ = run_matcher(exact=True, queries=["@@@"] + queries[:5], corpus=corpus)
    r = core.FusionMatcher(pool_size=5, device="cpu")
    install_fakes(r)
    out = r.link(["@@@"] + queries[:5], corpus, show_progress=False, details=True)
    pools = out["pools"]
    for p in pools:
        assert list(p) == sorted(p)
    assert len(pools[0]) == 10                   # no shared n-gram: 2k dense candidates
    assert all(len(p) <= 10 for p in pools)
    assert res["match_idx"][0] >= 0


def test_empty_inputs():
    queries, corpus = dataset(4)
    m = core.FusionMatcher(device="cpu")
    assert m.match([], corpus) == []
    assert m.match(["a"], []) == [(None, None)]


# --------------------------------------------------------------------------- index cache

def test_index_cache_miss_hit_and_key_changes(tmp_path):
    queries, corpus = dataset(5)
    first, m1, emb1, _ = run_matcher(exact=True, index_cache_dir=tmp_path, queries=queries, corpus=corpus)
    assert m1.retriever.last_index_timings["source"] == "computed"
    key = m1.retriever.last_index_timings["cache_key"]
    assert (tmp_path / key / "COMPLETE").exists()
    n_corpus_embedded = len(emb1.seen) - len(queries)

    # A second matcher (fresh process state) loads the index: the corpus is not embedded again.
    second, m2, emb2, _ = run_matcher(exact=True, index_cache_dir=tmp_path, queries=queries[::-1], corpus=corpus)
    assert m2.retriever.last_index_timings["source"] == "disk"
    assert len(emb2.seen) == len(queries) and n_corpus_embedded > 0
    same(second, run_matcher(exact=True, queries=queries[::-1], corpus=corpus)[0])
    assert np.array_equal(m1.retriever.corpus_embeddings, m2.retriever.corpus_embeddings)
    for a in ("data", "indices", "indptr"):
        assert np.array_equal(getattr(m1.retriever._tfidf_matrix, a), getattr(m2.retriever._tfidf_matrix, a))
    assert m2.retriever._tfidf_matrix.dtype == np.float32

    # Same matcher, same corpus again: reused from memory.
    m2.link(queries[:5], corpus, show_progress=False)
    assert m2.retriever.last_index_timings["source"] == "memory"

    # Any change of corpus text, order or batching mode is a different key (a miss).
    for kw in [dict(corpus=corpus[:-1] + ["NEW RECORD"]), dict(corpus=corpus[::-1]), dict(exact=False)]:
        args = dict(exact=True, index_cache_dir=tmp_path, queries=queries[:5], corpus=corpus)
        args.update(kw)
        _, m3, *_ = run_matcher(**args)
        assert m3.retriever.last_index_timings["source"] == "computed"
        assert m3.retriever.last_index_timings["cache_key"] != key


def test_hash_texts_is_unambiguous():
    assert index_cache.hash_texts(["ab", "c"]) != index_cache.hash_texts(["a", "bc"])
    assert index_cache.hash_texts(["a"]) != index_cache.hash_texts(["a", ""])
    assert index_cache.hash_texts(["x"] * 3) == index_cache.hash_texts(iter(["x"] * 3))
