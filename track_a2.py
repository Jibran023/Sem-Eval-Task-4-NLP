# import argparse
# import os
# import sys
# import time
# import random
# import pandas as pd
# from tqdm import tqdm
# from dotenv import load_dotenv

# # ---------- OpenAI client (lazy import so random baseline works without the package) ----------
# def make_openai_client(timeout_s: int = 30):
#     try:
#         from openai import OpenAI
#     except Exception as e:
#         sys.exit(f"[fatal] Missing OpenAI SDK. Install it with: pip install openai\n{e}")
#     key = os.getenv("OPENAI_API_KEY")
#     if not key:
#         sys.exit("[fatal] OPENAI_API_KEY not set. Export it or put it in a .env file.")
#     client = OpenAI()  # reads key from env
#     # Try to attach a per-request timeout
#     try:
#         return client.with_options(timeout=timeout_s)
#     except Exception:
#         # Older SDKs may not have with_options; we’ll just return the client.
#         return client

# # ---------- OpenAI pair scorer ----------
# def predict_pair_openai(client, model, anchor, A, B, temperature=0):
#     """
#     Returns True if A is closer; False if B is closer. Raises on hard errors.
#     """
#     resp = client.chat.completions.create(
#         model=model,
#         messages=[
#             {"role": "system",
#              "content": "You are an expert on narratives. Return only 'A' or 'B' for which is narratively closer to the anchor."},
#             {"role": "user",
#              "content": f"Anchor: {anchor}\n\nA: {A}\n\nB: {B}\n\nAnswer with only 'A' or 'B'."}
#         ],
#         temperature=temperature
#     )
#     ans = (resp.choices[0].message.content or "").strip()
#     if ans not in ("A", "B"):
#         # Be defensive; sometimes models add extra text
#         ans = "A" if " A" in ans or ans.startswith("A") else ("B" if " B" in ans or ans.startswith("B") else random.choice(["A","B"]))
#     return True if ans == "A" else False

# # ---------- Main runner ----------
# def run(args):
#     # Load env early, so OPENAI_API_KEY from .env works
#     load_dotenv()

#     if not os.path.exists(args.data):
#         sys.exit(f"[fatal] Data file not found: {args.data}")

#     df = pd.read_json(args.data, lines=True)
#     if args.max_rows and args.max_rows > 0:
#         df = df.head(args.max_rows)
#     print(f"[info] Loaded {len(df)} rows from {args.data}")

#     # Quick schema sanity check
#     required_cols = {"anchor_text", "text_a", "text_b"}
#     missing = required_cols - set(df.columns)
#     if missing:
#         sys.exit(f"[fatal] Missing required columns: {missing}. Your JSONL must include {sorted(required_cols)}.")

#     # Optional: gold labels for accuracy (if available)
#     has_gold = "text_a_is_closer" in df.columns

#     # Baseline selection
#     baseline = args.baseline.lower().strip()
#     if baseline not in {"openai", "random"}:
#         sys.exit("[fatal] --baseline must be one of: openai, random")

#     # If using OpenAI, prep the client
#     client = None
#     if baseline == "openai":
#         client = make_openai_client(timeout_s=args.timeout)
#         print(f"[info] Using OpenAI model: {args.model} (timeout={args.timeout}s, T={args.temperature})")

#     # --- tiny smoke test (first N rows) ---
#     N = min(args.smoke_test_rows, len(df))
#     if N > 0:
#         print(f"[info] Running a {N}-row smoke test...")
#         for i in range(N):
#             r = df.iloc[i]
#             print(f"  - smoke row {i+1}/{N}...", end="", flush=True)
#             try:
#                 if baseline == "openai":
#                     _ = predict_pair_openai(client, args.model, r["anchor_text"], r["text_a"], r["text_b"], temperature=args.temperature)
#                 else:
#                     _ = random.choice([True, False])
#                 print(" ok")
#             except Exception as e:
#                 print(f" error: {type(e).__name__}: {e}")
#                 if args.abort_on_smoke_fail:
#                     sys.exit("[fatal] Smoke test failed; fix the issue above and re-run.")
#         print("[info] Smoke test completed.\n")

#     # --- full run ---
#     preds = []
#     print(f"[info] Running predictions with baseline='{baseline}' on {len(df)} rows...")
#     for _, row in tqdm(df.iterrows(), total=len(df), dynamic_ncols=True):
#         try:
#             if baseline == "openai":
#                 pred = predict_pair_openai(client, args.model, row["anchor_text"], row["text_a"], row["text_b"], temperature=args.temperature)
#             else:
#                 pred = random.choice([True, False])
#         except Exception as e:
#             # Soft-fail with a random fallback so the script continues
#             pred = random.choice([True, False])
#         preds.append(pred)

#     df["predicted_text_a_is_closer"] = preds

#     # Accuracy (only if gold exists)
#     if has_gold:
#         acc = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
#         print(f"[info] Accuracy (vs gold): {acc:.3f}")
#     else:
#         print("[warn] No 'text_a_is_closer' column in data; skipping accuracy.")

#     # Write output JSONL in SemEval format (text_a_is_closer should contain the predictions)
#     os.makedirs(os.path.dirname(args.output), exist_ok=True) if os.path.dirname(args.output) else None
#     df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
#     df.drop(columns=["predicted_text_a_is_closer"], inplace=True)
#     with open(args.output, "w", encoding="utf-8") as f:
#         f.write(df.to_json(orient="records", lines=True))
#     print(f"[info] Wrote predictions to: {args.output}")

# def parse_args():
#     p = argparse.ArgumentParser(description="SemEval Track A baseline runner")
#     p.add_argument("--data", required=True, help="Path to input JSONL")
#     p.add_argument("--baseline", default="openai", choices=["openai","random"], help="Which baseline to run")
#     p.add_argument("--model", default="gpt-4o-mini", help="OpenAI model name")
#     p.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
#     p.add_argument("--timeout", type=int, default=30, help="Per-request timeout (seconds)")
#     p.add_argument("--max_rows", type=int, default=0, help="If >0, limit to first N rows")
#     p.add_argument("--smoke_test_rows", type=int, default=3, help="Run a tiny test before full inference")
#     p.add_argument("--abort_on_smoke_fail", action="store_true", help="Stop immediately if smoke test errors")
#     p.add_argument("--output", default="output/track_a.jsonl", help="Where to write predictions JSONL")
#     return p.parse_args()

# if __name__ == "__main__":
#     try:
#         run(parse_args())
#     except KeyboardInterrupt:
#         print("\n[cancelled] User interrupted.", file=sys.stderr)
#         sys.exit(1)











"""
Track A baseline system.

We use a naive prompt for chatGPT.

MODIFIED: Added tqdm for a progress bar.
"""

import random
from enum import Enum
import os

from openai import OpenAI
import pandas as pd
from pydantic import BaseModel
from tqdm import tqdm  # <-- 1. IMPORTED TQDM

# --- Configuration ---
BASELINE_TO_RUN = "openai"  # "openai" or "random"
INPUT_FILE = r"data\dev_track_a.jsonl" # <-- Used raw string
OUTPUT_FILE = r"output\track_a2.jsonl"   # <-- Used raw string

# --- Class Definitions ---

class ResponseEnum(str, Enum):
    A = "A"
    B = "B"


class SimilarityPrediction(BaseModel):
    explanation: str
    closer: ResponseEnum

# --- API Call Function ---

def predict(row):
    """
    Uses the OpenAI API to determine which of two stories (A or B) 
    is more narratively similar to an anchor story.

    Returns:
        bool: True if story A is predicted to be more similar; False otherwise.
    """
    anchor, text_a, text_b = row["anchor_text"], row["text_a"], row["text_b"]
    
    # Note: This will use the OPENAI_API_KEY from your environment variables
    completion = client.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": "You are an expert on stories and narratives. Tell us which of two stories is narratively similar to the anchor story.",
            },
            {
                "role": "user",
                "content": f"Anchor story: {anchor}\n\nStory A: {text_a}\n\nStory B: {text_b}",
            },
        ],
        response_format=SimilarityPrediction,
    )
    return completion.choices[0].message.parsed == ResponseEnum.A

# --- Main Execution ---

print(f"--- Running Baseline Script (Mode: {BASELINE_TO_RUN}) ---")
print(f"Loading data from: {INPUT_FILE}")

df = pd.read_json(INPUT_FILE, lines=True)

if BASELINE_TO_RUN == "openai":
    try:
        client = OpenAI()
        # Test if the client is authenticated before starting the loop
        client.models.list() 
    except Exception as e:
        print(f"❌ Error initializing OpenAI client: {e}")
        print("Please make sure your OPENAI_API_KEY environment variable is set correctly.")
        exit()

    print(f"Running OpenAI baseline on {len(df)} rows... (This will take 15-20 minutes)")
    
    tqdm.pandas(desc="Running OpenAI Baseline") # <-- 2. INITIALIZED TQDM
    
    # Use 'progress_apply' instead of 'apply'
    df["predicted_text_a_is_closer"] = df.progress_apply(predict, axis=1) # <-- 3. USED PROGRESS_APPLY
    
elif BASELINE_TO_RUN == "random":
    print(f"Running random baseline on {len(df)} rows...")
    df["predicted_text_a_is_closer"] = df.apply(
        lambda row: random.choice([True, False]), axis=1
    )

# --- Calculate and Print Accuracy ---
accuracy = (df["predicted_text_a_is_closer"] == df["text_a_is_closer"]).mean()
print(f"\n🎯🎯🎯 Baseline Accuracy: {accuracy:.3f} 🎯🎯🎯")


# --- Save Output ---
df["text_a_is_closer"] = df["predicted_text_a_is_closer"]
del df["predicted_text_a_is_closer"]

# Ensure output directory exists
os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)

print(f"Saving output to {OUTPUT_FILE}...")
open(OUTPUT_FILE, "w").write(df.to_json(orient='records', lines=True))

print("✅ Baseline script complete.")