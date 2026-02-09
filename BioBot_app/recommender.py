import os, re, json
import numpy as np
import pandas as pd
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from unidecode import unidecode

USE_FAISS = True
try:
    import faiss  # type: ignore
except Exception:
    USE_FAISS = False
    from sklearn.neighbors import NearestNeighbors

INDEX_DIR   = os.getenv("INDEX_DIR", "index")
CONFIG_PATH = os.path.join(INDEX_DIR, "config.json")
VECT_PATH   = os.path.join(INDEX_DIR, "vectors.npy")
META_PATH   = os.path.join(INDEX_DIR, "meta.parquet")
MODEL_CACHE = {}

def load_model(name: str):
    if name not in MODEL_CACHE:
        MODEL_CACHE[name] = SentenceTransformer(name)
    return MODEL_CACHE[name]

def tokenize(s: str) -> List[str]:
    s = (s or "").lower()
    s = unidecode(s)
    return re.findall(r"[a-z0-9]+", s)

def keyword_overlap_score(query: str, row: Dict) -> float:
    fields = [row.get("Ejemplos de plagas controladas",""), row.get("Ejemplos de aplicaciones",""),
              row.get("Uses",""), row.get("Description",""), row.get("Efficacy & activity","")]
    q = set(tokenize(query))
    doc = set(tokenize(" ".join([f for f in fields if f])))
    if not q or not doc:
        return 0.0
    inter = len(q & doc)
    return min(0.2, inter * 0.02)

class Recommender:
    def __init__(self, index_dir: str = INDEX_DIR):
        self.index_dir = index_dir
        self.config = json.load(open(CONFIG_PATH, "r", encoding="utf-8"))
        self.df = pd.read_parquet(META_PATH)
        self.vectors = np.load(VECT_PATH).astype(np.float32)
        self.model = load_model(self.config["model"])

        if USE_FAISS:
            dim = self.vectors.shape[1]
            self.index = faiss.IndexFlatIP(dim)
            self.index.add(self.vectors)
        else:
            self.index = NearestNeighbors(n_neighbors=10, metric="cosine").fit(self.vectors)

    def embed(self, text: str):
        return self.model.encode([text], normalize_embeddings=True)[0].astype(np.float32)

    def search(self, query: str, k: int = 5):
        v = self.embed(query)
        if USE_FAISS:
            D, I = self.index.search(np.expand_dims(v, 0), k)
            sims = D[0].tolist(); idxs = I[0].tolist()
        else:
            distances, indices = self.index.kneighbors([v], n_neighbors=k, return_distance=True)
            sims = [1 - float(d) for d in distances[0]]
            idxs = indices[0].tolist()

        rows = []
        for sim, i in zip(sims, idxs):
            row = self.df.iloc[int(i)].to_dict()
            rows.append((sim, row))

        rescored = []
        for sim, row in rows:
            sim2 = sim + keyword_overlap_score(query, row)
            rescored.append((sim2, row))
        rescored.sort(key=lambda x: x[0], reverse=True)
        return rescored[:k]

    def format_spanish(self, hit) -> str:
        sim, r = hit
        name = r.get("name","\n")
        pests = r.get("Example pests controlled","\n") or "—"
        apps  = r.get("Example applications","") or "—"
        uses  = r.get("Uses","\n") or "—"
        desc  = r.get("Description","\n") or "—"
        eff   = r.get("Efficacy & activity","\n") or "—"
        cite  = r.get("Please cite as","\n") or "—"
        return (
            f"◾ **{name}** (score: {sim:.2f})\n"
            f"• Descripción: {desc}\n"
            f"• Plagas controladas (ejemplos): {pests}\n"
            f"• Aplicaciones (ejemplos): {apps}\n"
            f"• Usos: {uses}\n"
            f"• Eficacia y actividad: {eff}\n"
            f"• Cita: {cite}"
        )
