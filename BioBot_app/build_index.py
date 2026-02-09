import json, os, re
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_PATH = os.getenv("DATA_PATH", "data/translated_biopesticides.csv")
OUT_DIR   = os.getenv("OUT_DIR", "index")
MODEL_NAME = os.getenv("MODEL_NAME", "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

# ========== Columnas esperadas (puedes editar si quieres) ==========
'''COLS_EN = [
    "name", "Description", "Example pests controlled", "Example applications",
    "Uses", "Efficacy & activity", "Canonical SMILES", "Isomeric SMILES", "Please cite as"
]'''
#COLS_ES = ["Descripción", "Ejemplos de plagas controladas", "Ejemplos de aplicaciones"],

COLS_ES = ['DescripciÃ³n', 'Ejemplos de plagas controladas', 'Ejemplos de aplicaciones', 'Usos', 'Eficacia y actividad']

    

# Asegura que no queden listas anidadas ni tipos raros en COLS_*
def _flatten_cols(seq):
    flat = []
    for x in seq:
        if isinstance(x, (list, tuple, set)):
            flat.extend([str(y) for y in x])
        else:
            flat.append(str(x))
    return flat
#COLS_EN = _flatten_cols(COLS_EN)
COLS_ES = _flatten_cols(COLS_ES)

os.makedirs(OUT_DIR, exist_ok=True)

def clean_text(x):
    if isinstance(x, (list, tuple, set)):
        # Si llega una lista (por ejemplo, ya tokenizado), la unimos
        return "; ".join(map(str, x))
    if not isinstance(x, str):
        x = "" if pd.isna(x) else str(x)
    x = x.replace("\xa0", " ").strip()
    x = re.sub(r"\s{2,}", " ", x)
    return x

def build_search_text(row: pd.Series) -> str:
    parts = []
    name = row.get("name", "")
    if isinstance(name, (list, tuple, set)):
        name = "; ".join(map(str, name))
    name = clean_text(name)
    if name:
        parts.append(f"Name: {name}")

    # Recorremos columnas esperadas en inglés y español SIN usar "c in row"
    for col in COLS_EN:
        val = row.get(col, "")
        val = clean_text(val)
        if val:
            parts.append(f"{col}: {val}")

    for col in COLS_ES:
        val = row.get(col, "")
        val = clean_text(val)
        if val:
            parts.append(f"{col}: {val}")

    return clean_text("\n".join(parts))

def main():
    # Lee CSV
    df = pd.read_csv(DATA_PATH)

    # Normaliza nombres de columnas (espacios/BOM)
    df.columns = [str(c).replace("\ufeff", "").strip() for c in df.columns]

    # Limpieza básica de todas las columnas tipo texto / lista
    for c in df.columns:
        df[c] = df[c].apply(clean_text)

    # Campo de búsqueda combinado (EN + ES si lo tienes)
    df["__search_text__"] = df.apply(build_search_text, axis=1)

    # Embeddings
    model = SentenceTransformer(MODEL_NAME)
    vectors = model.encode(
        df["__search_text__"].tolist(),
        normalize_embeddings=True,
        show_progress_bar=True
    ).astype(np.float32)

    # Guarda índice
    np.save(os.path.join(OUT_DIR, "vectors.npy"), vectors)
    df.drop(columns=["__search_text__"]).to_parquet(os.path.join(OUT_DIR, "meta.parquet"), index=False)
    with open(os.path.join(OUT_DIR, "config.json"), "w", encoding="utf-8") as f:
        json.dump(
            {"model": MODEL_NAME, "cols_en": COLS_EN, "cols_es": COLS_ES},
            f, ensure_ascii=False, indent=2
        )

    print("Index listo:", OUT_DIR, "| filas:", len(df), "| dim:", vectors.shape[1])

if __name__ == "__main__":
    main()
