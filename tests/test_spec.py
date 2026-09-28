"""The paper's specification (2026-09-27, "V5 + B2"): defaults, fusion rules and the B2 confidence.

When the replication archive's benchmark scripts are available (ENSEMBLELINK_BENCHMARK_SCRIPTS, default the
author's path), the package's fusion, top-1 and confidence are checked bit-for-bit against the benchmark's own
functions (ensemblelink_core.fuse, evaluate_linkage.make_predictions, ensemblelink_core.ensemblelink_confidence)
on random pools; those tests are skipped otherwise.
"""
import os
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import zeroshot_linkage as zl  # noqa: E402
from zeroshot_linkage import fusion, reranker  # noqa: E402

BENCH = Path(os.environ.get("ENSEMBLELINK_BENCHMARK_SCRIPTS",
                            "G:/Papers/submitted/combined/rr2/benchmark/scripts"))


def bench_modules():
    if not (BENCH / "ensemblelink_core.py").exists():
        pytest.skip("benchmark scripts not available")
    sys.path.insert(0, str(BENCH))
    for k in ["EL_RERANKERS", "EL_CONFIDENCE", "EL_RANK_FUSION", "EL_EXTRA_EXPERTS"]:
        os.environ.pop(k, None)
    import ensemblelink_core
    import evaluate_linkage
    if (getattr(ensemblelink_core, "DEFAULT_RERANKER_COLUMNS", None) != ("jina", "bge")
            or not hasattr(ensemblelink_core, "b2_confidence")):
        pytest.skip("benchmark scripts are not the 2026-09-27 specification")
    return ensemblelink_core, evaluate_linkage


# --------------------------------------------------------------------------- defaults

def test_default_models_and_licenses():
    assert zl.DEFAULT_RERANKER_MODELS == (reranker.JINA_RERANKER, reranker.BGE_RERANKER)
    assert zl.COMMERCIAL_RERANKER_MODELS == (reranker.BGE_RERANKER, reranker.ZERANK_RERANKER)
    assert reranker.MODEL_REVISIONS[reranker.JINA_RERANKER] == "9cfeff2df7d40d1b78e75e5e9cebec92a99813c9"
    assert reranker.MODEL_REVISIONS[reranker.BGE_RERANKER] == "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
    assert reranker.MODEL_REVISIONS["microsoft/harrier-oss-v1-0.6b"] == "f9b9dc8d367d443f2479d27aa5d8d2850c0774ee"
    m = zl.FusionMatcher(device="cpu")
    assert m.pool_size == 30 and m.csls_k == 10 and m.exact is True
    assert m.reranker.model_names == [reranker.JINA_RERANKER, reranker.BGE_RERANKER]
    assert m.reranker.batch_size == 128 and m.retriever.embed_max_length == 512


def test_precision_is_bfloat16_on_gpu_float32_on_cpu():
    import torch
    assert reranker.resolve_dtype("auto", "cuda") == torch.bfloat16
    assert reranker.resolve_dtype("auto", "cpu") == torch.float32
    assert reranker.resolve_dtype("float32", "cuda") == torch.float32


def test_multifield_prompt_and_record_text():
    import pandas as pd
    from zeroshot_linkage.linker import record_texts, MULTIFIELD_PROMPT
    assert MULTIFIELD_PROMPT == "Instruct: Retrieve the record that refers to the same entity\nQuery: "
    f = pd.DataFrame({"a": ["x ", None, "z"], "b": pd.array([1980, 1990, None], dtype="Int64")})
    assert record_texts(f, ["a", "b"], ["a", "b"]) == ["a=x | b=1980", "b=1990", "a=z"]
    assert record_texts(f, ["a"], ["a"]) == ["x", "", "z"]


# --------------------------------------------------------------------------- B2 (vectors of the benchmark's tests)

def test_b2_manual_values_with_average_ranks():
    got = fusion.b2_confidence(np.array([1.0, 1.0, 0.5, 0.0]), np.array([0.8, 0.8, 0.4, 0.99]),
                               np.array([False, False, False, True]))
    np.testing.assert_allclose(got, [(3.5 / 4 + 3.5 / 4) / 2, (3.5 / 4 + 3.5 / 4) / 2, (2 / 4 + 2 / 4) / 2, 0.0],
                               rtol=0, atol=0)


def test_b2_tied_probability_zeroed_before_ranking():
    margin = np.array([0.3, 0.2, 0.0]); tie = np.array([False, False, True])
    a = fusion.b2_confidence(margin, np.array([0.5, 0.6, 0.99]), tie)
    b = fusion.b2_confidence(margin, np.array([0.5, 0.6, 0.0]), tie)
    np.testing.assert_array_equal(a, b)
    assert a[2] == 0.0 and a[1] == pytest.approx((2 / 3 + 3 / 3) / 2)


def test_b2_missing_values_count_as_zero_and_order_equivariance():
    np.testing.assert_allclose(fusion.b2_confidence(np.array([np.nan, 0.4]), np.array([np.nan, 0.5]),
                                                    np.array([False, False])), [0.5, 1.0])
    rng = np.random.default_rng(3); m = rng.random(50); p = rng.random(50); t = rng.random(50) < .1
    perm = rng.permutation(50)
    np.testing.assert_array_equal(fusion.b2_confidence(m, p, t)[perm], fusion.b2_confidence(m[perm], p[perm], t[perm]))


def test_top_k_tie_rule():
    s = np.array([0.5, 0.9, 0.5, 0.5, 0.1, 0.9])
    assert fusion.top_k(s, 3).tolist() == [1, 5, 0]
    assert fusion.top_k(s, 4).tolist() == [1, 5, 0, 2]
    assert fusion.top_k(s, 10).tolist() == [1, 5, 0, 2, 3, 4]


def test_constant_expert_abstains():
    f = np.column_stack([[1.0, 0.0, -1.0], [0, 0, 0], [0, 1.0, -1.0], [0, 0, 0]])
    sel, mode = fusion.expert_votes("q", ["a", "b", "c"], f)
    assert sel.tolist() == [0, -1, 1, -1] and mode in (0, 1)


# --------------------------------------------------------------------------- against the benchmark's code

def random_unit(rng, n_queries=60, ties=True):
    """Pools with random expert inputs, some duplicated candidates (exact ties) and singleton pools."""
    import pandas as pd
    rows = []
    for qi in range(n_queries):
        q = f"q{qi:03d}"
        n = 1 if qi % 17 == 0 else int(rng.integers(2, 12))
        refs = sorted(rng.choice(200, n, replace=False))
        base = {k: rng.random(n).astype(np.float32) for k in ["jina", "bge", "dense", "sparse", "hub", "jw"]}
        if ties and n > 2 and qi % 5 == 0:     # an exact duplicate of the first candidate
            for v in base.values():
                v[1] = v[0]
        if qi % 7 == 0:                          # a constant (abstaining) lexical expert
            base["jw"][:] = 0.5
        for j, r in enumerate(refs):
            rows.append((q, f"r{r:04d}", float(base["dense"][j]), float(base["sparse"][j]), float(base["hub"][j]),
                         float(base["jw"][j]), base["jina"][j], base["bge"][j]))
    df = pd.DataFrame(rows, columns=["query_id", "reference_id", "dense", "sparse", "hub", "jw", "jina", "bge"])
    df["jina"] = df.jina.astype(np.float32); df["bge"] = df.bge.astype(np.float32)
    return df


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fusion_top1_and_b2_bit_identical_to_benchmark(seed):
    core, ev = bench_modules()
    rng = np.random.default_rng(seed)
    raw = random_unit(rng)
    qids = sorted(raw.query_id.unique()) + ["q_none"]           # one query without candidates
    scored, weights = core.fuse(raw)
    pred = ev.make_predictions(qids, scored[["query_id", "reference_id", "score"]])
    conf = core.ensemblelink_confidence(pred, scored, "b2")

    groups = {q: g.sort_values("reference_id") for q, g in raw.groupby("query_id")}
    feats, refs = [], []
    for q in qids:
        g = groups.get(q)
        if g is None:
            feats.append(None); refs.append([]); continue
        refs.append(g.reference_id.tolist())
        feats.append(fusion.expert_features([g.jina.to_numpy(), g.bge.to_numpy()], g.dense.to_numpy(), g.hub.to_numpy(),
                                            g["sparse"].to_numpy(), g.jw.to_numpy()))
    fused, w, _ = fusion.fuse(qids, refs, feats)
    np.testing.assert_array_equal(w, np.array(weights["weights"]))
    margin, prob, tie, win = [], [], [], []
    for q, r, f in zip(qids, refs, fused):
        if f is None:
            margin.append(np.nan); prob.append(np.nan); tie.append(False); win.append(None); continue
        np.testing.assert_array_equal(f, scored[scored.query_id == q].sort_values("reference_id").score.to_numpy())
        i, best, mg, nt = fusion.select_top(q, r, f)
        g = groups[q]
        win.append(r[i]); margin.append(mg); tie.append(nt > 1)
        prob.append(fusion.mean_probability([np.array([g.jina.iloc[i]]), np.array([g.bge.iloc[i]])])[0])
    assert win == pred.reference_id.tolist()
    np.testing.assert_array_equal(np.array(margin), pred.margin.to_numpy(float))
    got = fusion.b2_confidence(np.array(margin), np.array(prob), np.array(tie))
    np.testing.assert_array_equal(got, conf)
