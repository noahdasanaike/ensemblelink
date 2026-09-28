"""Copy the Python package into the R package's backend (r_package/inst/python/ensemblelink_py/).

The R backend runs the same code as the Python package; tests/test_r_backend.py fails if the copy is stale.

  python tools/sync_r_backend.py
"""
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "zeroshot_linkage"
DST = ROOT / "r_package" / "inst" / "python" / "ensemblelink_py"
MODULES = ["__init__.py", "core.py", "fusion.py", "retrieval.py", "reranker.py", "linker.py", "index_cache.py",
           "_fast_tfidf.py", "occupations.py"]


def main():
    if DST.exists():
        shutil.rmtree(DST)
    DST.mkdir(parents=True)
    for m in MODULES:
        shutil.copyfile(SRC / m, DST / m)
    print("copied", len(MODULES), "modules to", DST)


if __name__ == "__main__":
    main()
