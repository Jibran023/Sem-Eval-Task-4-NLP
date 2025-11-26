import os
os.environ["TRANSFORMERS_NO_TORCHVISION"] = "1" 

import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
from sentence_transformers.util import cos_sim

# 1. Load datasets
print("\n[Stage 1/5] Loading data...")

track_a_path = "data/dev_track_a.jsonl"
track_b_path = "data/dev_track_b.jsonl"

# Load Track A (triplets) and Track B (story corpus)
df_a = pd.read_json(track_a_path, lines=True)
df_b = pd.read_json(track_b_path, lines=True)
print(f"[info] ✅ Loaded {len(df_a)} Track A triplets and {len(df_b)} Track B stories")


# 2. Load pretrained model
print("\n[Stage 2/5] Loading pretrained model...")

model_name = "intfloat/e5-large-v2"  # strong general embedding model
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[info] Using model: {model_name} on {device}")

# Load SentenceTransformer model
model = SentenceTransformer(model_name, device=device)

def embed(texts):
    """Generate normalized embeddings using the E5 'passage:' prompt format."""
    print(f"[info] Encoding {len(texts)} texts ...")
    return model.encode(
        [f"passage: {t.strip()}" for t in texts],
        normalize_embeddings=True,   # ensures cosine sim works directly
        convert_to_numpy=True,
        show_progress_bar=True
    )


# 3. Encode Track B corpus (used for lookup)
print("\n[Stage 3/5] Encoding Track-B corpus...")

# All text content from Track B
texts = df_b["text"].astype(str).tolist()

# Encode and normalize embeddings
embeddings = embed(texts)
print(f"[info] ✅ Encoded Track-B corpus → shape {embeddings.shape}")

# Quick lookup map from text → embedding
emb_lookup = dict(zip(texts, embeddings))


# 4. Evaluate Track A triplets via cosine similarity
print("\n[Stage 4/5] Evaluating cosine-similarity baseline...")

correct = 0
total = 0
missing = 0

# Iterate through each triplet (anchor, A, B)
for row in tqdm(df_a.itertuples(), total=len(df_a), desc="Evaluating triplets"):
    v_anchor = emb_lookup.get(row.anchor_text)
    v_a = emb_lookup.get(row.text_a)
    v_b = emb_lookup.get(row.text_b)

    # skip triplets if any text is missing in corpus
    if v_anchor is None or v_a is None or v_b is None:
        missing += 1
        continue

    # Compute cosine similarity between anchor and each candidate
    sim_a = cos_sim(v_anchor, v_a)
    sim_b = cos_sim(v_anchor, v_b)

    # Pick whichever is closer (higher cosine)
    pred = sim_a > sim_b

    # Compare with ground truth
    if pred.item() == row.text_a_is_closer:
        correct += 1
    total += 1

# Compute simple accuracy
accuracy = correct / total
print(f"\n[info] ✅ Evaluation complete. Skipped {missing} triplets (missing embeddings).")
print(f"🎯 Final Accuracy: {accuracy * 100:.2f}% on {total} valid triplets")


# 5. Save encoded embeddings for reuse
os.makedirs("output", exist_ok=True)
np.save("output/e5_trackb_embeddings.npy", embeddings)
print("[info] ✅ Saved Track B embeddings → output/e5_trackb_embeddings.npy")

print("\n✨ Done — baseline uses pure cosine similarity, no classifier involved.")
