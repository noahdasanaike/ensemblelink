"""The R package's Python backend (r_package/inst/python/matcher.py) runs the vendored Python package.

Checks that the vendored copy (ensemblelink_py) is identical to zeroshot_linkage (run
tools/sync_r_backend.py after changing the package) and that the R-facing matcher returns the
Python package's results, with fake models.
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from test_speedups import FakeEmbedder, fake_pair_score, dataset, run_matcher  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
VENDORED = ROOT / "r_package" / "inst" / "python" / "ensemblelink_py"


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_vendored_package_is_current():
    sync = load(ROOT / "tools" / "sync_r_backend.py", "el_sync")
    assert sorted(p.name for p in VENDORED.glob("*.py")) == sorted(sync.MODULES)
    for m in sync.MODULES:
        assert (VENDORED / m).read_bytes() == (ROOT / "zeroshot_linkage" / m).read_bytes(), m + " is stale"


def fake(matcher):
    m = matcher._m
    m.retriever._embed_model = FakeEmbedder()

    def score_pairs(pairs, show_progress=False, sort_by_length=False, **kw):
        return [np.array([fake_pair_score(q, c, s) for q, c in pairs], dtype=np.float32) for s in (1, 2)]

    m.reranker.score_pairs = score_pairs


def test_r_matcher_equals_python_package(tmp_path):
    R = load(ROOT / "r_package" / "inst" / "python" / "matcher.py", "el_r_matcher")
    queries, corpus = dataset(7)
    for exact in (True, False):
        m = R.EnsembleMatcher(pool_size=5, device="cpu", exact=exact, index_cache=str(tmp_path))
        fake(m)
        m.index(corpus, show_progress=False)
        got = m.match(queries, return_scores=True, show_progress=False)
        ref, *_ = run_matcher(exact=exact, queries=queries, corpus=corpus)
        assert got["indices"] == [int(i) for i in ref["match_idx"]]
        np.testing.assert_array_equal(np.array(got["score"], dtype=float), ref["score"])
        assert got["matches"] == [corpus[i] for i in ref["match_idx"]]
        assert m.match(queries[:3], show_progress=False) == got["matches"][:3]


def test_r_blocked_matcher_runs():
    R = load(ROOT / "r_package" / "inst" / "python" / "matcher.py", "el_r_matcher2")
    b = R.BlockedMatcher(pool_size=5, device="cpu")
    fake(b._matcher)
    blocks = ["CA", "CA", "TX", "TX", "NY"]
    details = ["Los Angeles", "San Francisco", "Harris", "Dallas", "Queens"]
    b.index(blocks, details, show_progress=False)
    res = b.match(["Kalifornia", "Texass"], ["Los Angelos", "Harris Co"], return_scores=True, show_progress=False)
    assert len(res["match_indices"]) == 2 and all(i is not None for i in res["match_indices"])
