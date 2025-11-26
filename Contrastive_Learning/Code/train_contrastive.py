import json
import torch
import os
import sys
import pandas as pd
from sentence_transformers import SentenceTransformer, losses
from sentence_transformers.readers import InputExample
from torch.utils.data import DataLoader
from typing import Dict, List, Any
# Import the evaluator
from sentence_transformers.evaluation import TripletEvaluator

# <--- FIX 1: Add this back to prevent the code from hanging ---
os.environ["WANDB_DISABLED"] = "true"

# --- Configuration ---

# 1. SET YOUR FOLDER AND FILE NAMES
BASE_DIR = 'Contrastive_Learning/Data'
ORIGINAL_DATA_FILE = os.path.normpath(os.path.join(BASE_DIR, 'contrastive_learning_dataset.jsonl'))
AUGMENTED_DATA_FILE = os.path.normpath(os.path.join(BASE_DIR, 'augmented_contrastive.jsonl'))

# Path to the dev set for evaluation
TRACK_A_DEV_FILE = 'data/dev_track_a.jsonl' 

# 2. SET MODEL PARAMETERS
BASE_MODEL = 'intfloat/e5-large-v2'
FINETUNED_MODEL_PATH = './fine-tuned-e5-narrative'

# 3. SET TRAINING PARAMETERS
BATCH_SIZE = 4 # Try 4. If it still crashes, try 2 or 1.
GRADIENT_ACCUMULATION_STEPS = 4 
NUM_EPOCHS = 1
TRIPLET_MARGIN = 1.0

# --- End Configuration ---


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    # <--- ADDED PRINT ---
    print(f"  [IO] Loading data from {filepath}...")
    data = []
    if not os.path.exists(filepath):
        print(f"  [IO] Warning: File not found at {filepath}, returning empty list.")
        return data

    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"  [IO] Warning: Skipping malformed line in {filepath}")
    except Exception as e:
        print(f"  [IO] Error: Could not read file {filepath}. Error: {e}")
    
    # <--- ADDED PRINT ---
    print(f"  [IO] Loaded {len(data)} lines.")
    return data

def create_training_examples(data: List[Dict[str, Any]]) -> List[InputExample]:
    """
    Converts the JSONL data into InputExamples for TripletLoss.
    We must add the 'passage: ' prefix for E5 models.
    """
    # <--- ADDED PRINT ---
    print("  [Data] Creating training examples (adding 'passage:' prefix)...")
    examples = []
    for item in data:
        anchor_key = 'anchor_story' if 'anchor_story' in item else 'anchor_text'

        anchor = item.get(anchor_key)
        positive = item.get('similar_story')
        negative = item.get('dissimilar_story')

        if not all([anchor, positive, negative]):
            # print("Warning: Skipping item with missing keys.") # This can be too noisy, commenting out
            continue

        # Add the 'passage: ' prefix required by E5 models
        anchor = f"passage: {anchor}"
        positive = f"passage: {positive}"
        negative = f"passage: {negative}"

        examples.append(InputExample(texts=[anchor, positive, negative]))

    # <--- ADDED PRINT ---
    print(f"  [Data] Created {len(examples)} training examples from this file.")
    return examples

def load_track_a_dev_data(filepath: str) -> List[InputExample]:
    """
    Loads the Track A dev set and converts it to (anchor, positive, negative)
    triplets for the TripletEvaluator.
    """
    # <--- ADDED PRINT ---
    print(f"  [Data] Loading evaluation data from {filepath}...")
    dev_examples = []
    data = load_jsonl(filepath) # This will call the modified load_jsonl with its prints
    
    if not data:
        print(f"  [Data] Warning: No evaluation data found at {filepath}.")
        return []

    # <--- ADDED PRINT ---
    print("  [Data] Creating evaluation triplets from Track A data...")
    for item in data:
        anchor = item.get('anchor_text')
        text_a = item.get('text_a')
        text_b = item.get('text_b')
        a_is_closer = item.get('text_a_is_closer')

        if not all([anchor, text_a, text_b, a_is_closer is not None]):
            print("  [Data] Warning: Skipping dev item with missing keys.")
            continue
            
        # Add the 'passage: ' prefix required by E5 models
        anchor = f"passage: {anchor}"
        
        if a_is_closer:
            # A is the positive, B is the negative
            positive = f"passage: {text_a}"
            negative = f"passage: {text_b}"
        else:
            # B is the positive, A is the negative
            positive = f"passage: {text_b}"
            negative = f"passage: {text_a}"
            
        dev_examples.append(InputExample(texts=[anchor, positive, negative]))
        
    # <--- ADDED PRINT ---
    print(f"  [Data] Created {len(dev_examples)} evaluation examples.")
    return dev_examples

def main():
    # --- 1. Setup Device ---
    # <--- ADDED PRINTS ---
    print("\n--- 1. Setting up Device ---")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Using device: {device.upper()}")
    if device == "cpu":
        print("  WARNING: GPU not found. Training will be VERY slow.")
    print("---------------------------------")

    # --- 2. Load Base Model ---
    # <--- ADDED PRINTS ---
    print(f"\n--- 2. Loading Base Model ---")
    print(f"  Loading {BASE_MODEL}...")
    model = SentenceTransformer(BASE_MODEL, device=device)
    model.max_seq_length = 512
    print(f"  ✅ Model loaded. Max sequence length set to {model.max_seq_length}.")
    print("-----------------------------")

    # --- 3. Load and Combine Data ---
    # <--- ADDED PRINTS ---
    print(f"\n--- 3. Loading Training Data ---")
    original_data = load_jsonl(ORIGINAL_DATA_FILE)
    augmented_data = load_jsonl(AUGMENTED_DATA_FILE)

    if not original_data and not augmented_data:
        print(f"  [Error] No data found in {ORIGINAL_DATA_FILE} or {AUGMENTED_DATA_FILE}.")
        sys.exit(1)

    print("\n  [Data] Processing training data...")
    train_examples_original = create_training_examples(original_data)
    train_examples_augmented = create_training_examples(augmented_data)

    train_examples = train_examples_original + train_examples_augmented
    print(f"  ✅ Total training examples: {len(train_examples)}")

    if not train_examples:
        print("  [Error] No valid training examples were created. Check your data files.")
        sys.exit(1)
    print("--------------------------------")

    # --- 4. Prepare Dataloader and Loss ---
    # <--- ADDED PRINTS ---
    print(f"\n--- 4. Preparing Dataloader & Loss ---")
    train_dataloader = DataLoader(train_examples, shuffle=True, batch_size=BATCH_SIZE)
    print(f"  Dataloader created with batch size: {BATCH_SIZE}")

    train_loss = losses.TripletLoss(
        model=model,
        distance_metric=losses.TripletDistanceMetric.COSINE,
        triplet_margin=TRIPLET_MARGIN
    )
    print(f"  TripletLoss configured with margin: {TRIPLET_MARGIN}")
    print("--------------------------------------")

    # --- 5. Prepare Evaluator (to fight overfitting) ---
    # <--- ADDED PRINTS ---
    print(f"\n--- 5. Preparing Evaluator ---")
    dev_examples = load_track_a_dev_data(TRACK_A_DEV_FILE)
    evaluator = None
    evaluation_steps = 0
    
    if dev_examples:
        evaluator = TripletEvaluator.from_input_examples(
            dev_examples, 
            name='dev-triplets'
        )
        # Calculate evaluation steps
        evaluation_steps = 250 # Evaluate every 250 steps
        print(f"  ✅ Evaluator created. Will test on {len(dev_examples)} dev examples every {evaluation_steps} steps.")
    else:
        print("  [Warn] No evaluation data loaded. Will not run evaluation during training.")
    print("------------------------------")


    # --- 6. Start Training ---
    # <--- ADDED PRINTS ---
    print(f"\n--- 6. Starting Fine-Tuning ---")
    print(f"  Base Model: {BASE_MODEL}")
    print(f"  Epochs: {NUM_EPOCHS}")
    print(f"  Batch Size (per device): {BATCH_SIZE}")
    print(f"  Gradient Accumulation Steps: {GRADIENT_ACCUMULATION_STEPS}")
    print(f"  EFFECTIVE Batch Size: {BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS}")
    print(f"  Use Automatic Mixed Precision (AMP): True")
    print(f"  Will save best model to: {FINETUNED_MODEL_PATH}")
    print("\n▶️  --- RUNNING MODEL.FIT() [Progress bar will appear below] ---")

    model.fit(
        train_objectives=[(train_dataloader, train_loss)],
        epochs=NUM_EPOCHS,
        warmup_steps=100,
        output_path=FINETUNED_MODEL_PATH,
        show_progress_bar=True,
        
        # The evaluator you created to monitor performance
        evaluator=evaluator,
        
        # How often to run the evaluator
        evaluation_steps=evaluation_steps, 
        
        # This saves the *best* model, not the last one
        save_best_model=True, 
        
        # gradient_accumulation_steps (will work after you upgrade)
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS 
    )

    # <--- ADDED PRINTS ---
    print("\n🏁 --- Training complete ---")
    print(f"  ✅ Best fine-tuned model saved to: {FINETUNED_MODEL_PATH}")
    print("-----------------------------")

if __name__ == "__main__":
    main()