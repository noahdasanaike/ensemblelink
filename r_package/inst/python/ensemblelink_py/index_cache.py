"""
On-disk cache of a corpus index (embeddings, TF-IDF matrix, fitted vectorizer).

Linking several query batches against one large corpus re-embeds the corpus on
every call, and embedding is the slowest step. With ``index_cache`` set, the
first call saves the index under a key and later calls load it instead.

The key is a SHA-256 over everything that can change a single float of the
index: the corpus texts (in order), the embedding model id and resolved Hugging
Face revision, the batching mode and sizes, the TF-IDF settings, the versions of
the libraries that compute the index, and the device (GPU model) the embeddings
were computed on. Any change gives a new key, so a hit is exactly the index that
would have been recomputed. Entries are written to a temporary directory and
renamed into place, so an interrupted write never leaves a partial entry.

The fitted vectorizer is stored with pickle: point ``index_cache`` only at a
directory you trust.
"""

import hashlib
import json
import os
import pickle
import shutil
import tempfile
import time
from typing import Iterable, Optional

import numpy as np

FORMAT_VERSION = "ensemblelink-index-v1"


def hash_texts(texts: Iterable[str]) -> str:
    """SHA-256 of an ordered list of strings (length-prefixed, so no separator ambiguity)."""
    h = hashlib.sha256()
    n = 0
    buf = []
    size = 0
    for t in texts:
        b = t.encode("utf-8", "surrogatepass")
        buf.append(len(b).to_bytes(8, "little"))
        buf.append(b)
        size += len(b) + 8
        n += 1
        if size > (1 << 22):
            h.update(b"".join(buf))
            buf, size = [], 0
    h.update(b"".join(buf))
    h.update(n.to_bytes(8, "little"))
    return h.hexdigest()


def resolve_model_revision(model_name: str, cache_dir: Optional[str] = None) -> str:
    """Commit hash of a cached Hugging Face model, or a content hash of a local model directory."""
    if os.path.isdir(model_name):
        h = hashlib.sha256()
        for root, _, files in sorted(os.walk(model_name)):
            for f in sorted(files):
                p = os.path.join(root, f)
                st = os.stat(p)
                h.update(f"{os.path.relpath(p, model_name)}:{st.st_size}:{int(st.st_mtime)}".encode())
        return "local:" + h.hexdigest()
    try:
        from huggingface_hub import try_to_load_from_cache

        for fname in ("config.json", "modules.json", "config_sentence_transformers.json"):
            for cd in (cache_dir, None):
                path = try_to_load_from_cache(model_name, fname, cache_dir=cd)
                if isinstance(path, str):
                    parts = os.path.normpath(path).split(os.sep)
                    if "snapshots" in parts:
                        return parts[parts.index("snapshots") + 1]
    except Exception:
        pass
    return "unresolved"


def library_versions() -> dict:
    out = {}
    for mod in ("numpy", "scipy", "sklearn", "torch", "transformers", "sentence_transformers", "unidecode"):
        try:
            m = __import__(mod)
            out[mod] = getattr(m, "__version__", "unknown")
        except Exception:
            out[mod] = None
    return out


def device_tag(device: Optional[str]) -> str:
    try:
        import torch

        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        if str(device).startswith("cuda") and torch.cuda.is_available():
            return f"{device}:{torch.cuda.get_device_name(torch.device(device))}"
        return str(device)
    except Exception:
        return str(device)


def make_key(components: dict) -> str:
    blob = json.dumps({"format": FORMAT_VERSION, **components}, sort_keys=True, default=str)
    return hashlib.sha256(blob.encode()).hexdigest()[:32]


def load(cache_root: str, key: str):
    """Return ``(embeddings, tfidf_matrix, vectorizer)`` for ``key``, or ``None`` on a miss."""
    d = os.path.join(cache_root, key)
    if not os.path.isfile(os.path.join(d, "COMPLETE")):
        return None
    import scipy.sparse as sp

    emb = np.load(os.path.join(d, "embeddings.npy"))
    with np.load(os.path.join(d, "tfidf.npz")) as z:
        mat = sp.csr_matrix((z["data"], z["indices"], z["indptr"]), shape=tuple(z["shape"]))
    with open(os.path.join(d, "vectorizer.pkl"), "rb") as f:
        vec = pickle.load(f)
    return emb, mat, vec


def save(cache_root: str, key: str, components: dict, embeddings, tfidf_matrix, vectorizer) -> str:
    """Write an entry atomically; returns its directory. An existing entry is left as is."""
    os.makedirs(cache_root, exist_ok=True)
    final = os.path.join(cache_root, key)
    if os.path.isfile(os.path.join(final, "COMPLETE")):
        return final
    tmp = tempfile.mkdtemp(prefix=f".{key}.", dir=cache_root)
    try:
        np.save(os.path.join(tmp, "embeddings.npy"), np.ascontiguousarray(embeddings))
        # Raw CSR arrays, stored as is (no re-sorting), so a load is bit-identical.
        np.savez(os.path.join(tmp, "tfidf.npz"), data=tfidf_matrix.data, indices=tfidf_matrix.indices,
                 indptr=tfidf_matrix.indptr, shape=np.array(tfidf_matrix.shape))
        with open(os.path.join(tmp, "vectorizer.pkl"), "wb") as f:
            pickle.dump(vectorizer, f, protocol=pickle.HIGHEST_PROTOCOL)
        with open(os.path.join(tmp, "meta.json"), "w") as f:
            json.dump({"format": FORMAT_VERSION, "key": key, "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
                       "n_records": int(embeddings.shape[0]), **components}, f, indent=2, sort_keys=True, default=str)
        open(os.path.join(tmp, "COMPLETE"), "w").close()
        if os.path.isdir(final) and not os.path.isfile(os.path.join(final, "COMPLETE")):
            shutil.rmtree(final, ignore_errors=True)  # debris without a COMPLETE marker
        try:
            os.replace(tmp, final)
        except OSError:
            # Another process wrote the same entry first (or the target exists): keep theirs.
            shutil.rmtree(tmp, ignore_errors=True)
    except BaseException:
        shutil.rmtree(tmp, ignore_errors=True)
        raise
    return final
