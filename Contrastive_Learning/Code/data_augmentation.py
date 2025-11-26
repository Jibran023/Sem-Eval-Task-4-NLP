#!/usr/bin/env python3
import json
import os
import sys
import torch
import nltk # <-- Added NLTK for sentence splitting
from nltk.tokenize import sent_tokenize
from transformers import MarianMTModel, MarianTokenizer
from tqdm import tqdm
from typing import Dict, List, Any, Optional

# --- Configuration for VS Code ---

# 1. SET YOUR FOLDER AND FILE NAMES
# Set the base directory using forward slashes
BASE_DIR = 'Contrastive_Learning/Data'

# Set the file names
INPUT_NAME = 'contrastive_learning_dataset.jsonl'
OUTPUT_NAME = 'augmented_contrastive.jsonl'
CHECKPOINT_NAME = 'contrastive_checkpoint.json'

# --- Automatically build OS-safe paths ---
# This uses os.path.join to create paths that work on any OS (Windows, Mac, Linux)
INPUT_FILE = os.path.normpath(os.path.join(BASE_DIR, INPUT_NAME))
OUTPUT_FILE = os.path.normpath(os.path.join(BASE_DIR, OUTPUT_NAME))
CHECKPOINT_FILE = os.path.normpath(os.path.join(BASE_DIR, CHECKPOINT_NAME))
# -------------------------------------------

# Example for your second run:
# BASE_DIR = 'Contrastive_Learning/Data'
# INPUT_NAME = 'classification_data.jsonl'
# OUTPUT_NAME = 'augmented_classification.jsonl'
# CHECKPOINT_NAME = 'classification_checkpoint.json'


# 2. SET RESTART FLAG
# True: Deletes old OUTPUT_FILE and CHECKPOINT_FILE and starts from row 0.
# False: Looks for CHECKPOINT_FILE and resumes where it left off.
FORCE_RESTART = False # <-- Set to False to resume your previous run

# 3. SET LANGUAGES
# Based on logs, 'de' and 'ru' had quality issues ("Wanderer" -> "Hikers").
# 'fr', 'es', 'zh' had truncation/failure issues.
# 'it' and 'pt' are good candidates.
LANGUAGES = ['fr', 'es', 'nl', 'it']
# ------------------------------------


# --- Helper Functions for Data I/O and Checkpointing ---

def setup_nltk():
    """Downloads the NLTK 'punkt' tokenizer if not found."""
    try:
        nltk.data.find('tokenizers/punkt')
    except LookupError:
        print("Downloading NLTK 'punkt' tokenizer...")
        nltk.download('punkt')
    print("NLTK 'punkt' tokenizer is ready.")


def load_jsonl(filepath: str) -> List[Dict[str, Any]]:
    """Loads a JSONL file into a list of dictionaries."""
    data = []
    if not os.path.exists(filepath):
        print(f"Error: File not found at {filepath}, returning empty list.")
        sys.exit(1)
        
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError:
                    print(f"Warning: Skipping malformed line in {filepath}")
    except Exception as e:
        print(f"Error: Could not read file {filepath}. Error: {e}")
        sys.exit(1)
    return data

def append_jsonl(filepath: str, data_item: Dict[str, Any]):
    """Appends a single item to a JSONL file."""
    try:
        # --- FIX: Ensure directory exists before writing ---
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # --------------------------------------------------
        with open(filepath, 'a', encoding='utf-8') as f:
            f.write(json.dumps(data_item) + '\n')
    except Exception as e:
        print(f"Error: Could not append to file {filepath}. Error: {e}")

def load_state(state_filepath: str) -> int:
    """Loads the last processed index from the state file."""
    if not os.path.exists(state_filepath):
        return -1 # Start from the beginning
    try:
        with open(state_filepath, 'r', encoding='utf-8') as f:
            state = json.load(f)
            return state.get("last_processed_index", -1)
    except Exception as e:
        print(f"Warning: Could not load state from {state_filepath}. Resetting. Error: {e}")
        return -1

def save_state(state_filepath: str, index: int):
    """Saves the last processed index to the state file."""
    try:
        # --- FIX: Ensure directory exists before writing ---
        os.makedirs(os.path.dirname(state_filepath), exist_ok=True)
        # --------------------------------------------------
        with open(state_filepath, 'w', encoding='utf-8') as f:
            json.dump({"last_processed_index": index}, f)
    except Exception as e:
        print(f"Error: Could not save state to {state_filepath}. Error: {e}")


# --- Translation Functions ---

def load_translation_models(languages, device):
    """
    Loads all required translation models (EN->X and X->EN) into memory.
    """
    models = {}
    print("--- Loading Translation Models ---")
    for lang in languages:
        try:
            # Load EN -> Language model
            model_name_en_x = f'Helsinki-NLP/opus-mt-en-{lang}'
            print(f"Loading model: {model_name_en_x}...")
            models[f'en_to_{lang}'] = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name_en_x),
                'model': MarianMTModel.from_pretrained(model_name_en_x).to(device)
            }
            
            # Load Language -> EN model
            model_name_x_en = f'Helsinki-NLP/opus-mt-{lang}-en'
            print(f"Loading model: {model_name_x_en}...")
            models[f'{lang}_to_en'] = {
                'tokenizer': MarianTokenizer.from_pretrained(model_name_x_en),
                'model': MarianMTModel.from_pretrained(model_name_x_en).to(device)
            }
        except OSError as e:
            print(f"Warning: Could not load models for language '{lang}'. Skipping. Error: {e}")
            if f'en_to_{lang}' in models:
                del models[f'en_to_{lang}']
    
    # Filter out any languages that failed to load completely
    loaded_languages = [lang for lang in languages if f'en_to_{lang}' in models and f'{lang}_to_en' in models]
    print(f"\nSuccessfully loaded models for: {loaded_languages}")
    return models, loaded_languages


def translate_single_chunk(text_chunk: str, model, tokenizer, device, max_length=512) -> Optional[str]:
    """
    Translates a *single chunk* of text (e.g., a sentence).
    This function WILL truncate if the chunk is > max_length.
    """
    try:
        # Tokenize the text
        inputs = tokenizer(text_chunk, return_tensors="pt", padding=True, truncation=True, max_length=max_length).to(device)
        
        # Generate translation
        translated_ids = model.generate(inputs.input_ids, max_length=max_length, num_beams=4, early_stopping=True)
        
        # Decode and return the translated text
        translated_text = tokenizer.batch_decode(translated_ids, skip_special_tokens=True)[0]
        return translated_text
    except Exception as e:
        print(f"  > Error during single chunk translation: {e}")
        return None

def translate_full_text(text: str, model, tokenizer, device, progress_bar, log_prefix=""):
    """
    Translates a full-length story by splitting it into sentences
    to avoid the 512-token truncation limit.
    """
    try:
        # Split the full text into individual sentences
        sentences = sent_tokenize(text)
        translated_sentences = []
        
        # Translate sentence by sentence (safest way to avoid truncation)
        for sentence in sentences:
            if not sentence.strip():
                continue
            
            # Use the original translate function for each chunk
            translated_chunk = translate_single_chunk(sentence, model, tokenizer, device)
            
            # --- CRITICAL FIX ---
            # If any chunk fails, abort the *entire* translation for this language
            if not translated_chunk:
                progress_bar.write(f"{log_prefix} WARNING: Failed to translate chunk: {sentence[:30]}...")
                progress_bar.write(f"{log_prefix} Aborting translation for this language.")
                return None # <-- Return None to signal failure
            # --------------------
            
            translated_sentences.append(translated_chunk)
        
        # Join the translated sentences back together
        return " ".join(translated_sentences)
    
    except Exception as e:
        progress_bar.write(f"{log_prefix} ERROR in translate_full_text: {e}")
        return None

# --- Main Function ---

def main():
    # Check for GPU
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("-------------------------------------------------")
    print(f"--- Running on device: {device.upper()} ---")
    if device == "cpu":
        print("--- WARNING: GPU not found. Back-translation will be VERY slow.")
    print("-------------------------------------------------")

    # --- NEW: Setup NLTK ---
    setup_nltk()
    print("-------------------------------------------------")
    
    # --- Handle Force Restart ---
    if FORCE_RESTART:
        print("FORCE_RESTART is True. Deleting old output and state files...")
        files_to_delete = [OUTPUT_FILE, CHECKPOINT_FILE]
        for f_path in files_to_delete:
            if os.path.exists(f_path):
                try:
                    os.remove(f_path)
                    print(f"  > Deleted {f_path}")
                except Exception as e:
                    print(f"  > Warning: Could not delete {f_path}. Error: {e}")
        print("Reset complete. Starting from row 0.")

    # 1. Load all models
    models, loaded_languages = load_translation_models(LANGUAGES, device)
    if not loaded_languages:
        print("No models were loaded. Exiting.")
        sys.exit(1)
    
    # 2. Load the full input dataset
    print(f"Loading data from {INPUT_FILE}...")
    input_data = load_jsonl(INPUT_FILE)
    if not input_data:
        print(f"Error: No data found in {INPUT_FILE}. Aborting.")
        return

    # 3. Load state to find out where we left off
    last_processed_index = load_state(CHECKPOINT_FILE)
    start_index = last_processed_index + 1

    if start_index >= len(input_data):
        print("All items already processed. Augmentation complete.")
        return
    elif start_index > 0:
        print(f"Resuming augmentation from item {start_index}...")

    # 4. Start augmentation loop
    print(f"\n--- Starting augmentation process for {len(input_data) - start_index} items ---")
    completed_count_session = 0
    
    try:
        # --- MODIFIED: Store the tqdm bar in a variable ---
        progress_bar = tqdm(range(start_index, len(input_data)), desc="Processing items")
        
        for i in progress_bar:
            data = input_data[i]
            
            # Find the anchor key (flexible for both your files)
            anchor_key = None
            if 'anchor_story' in data:
                anchor_key = 'anchor_story'
            elif 'anchor_text' in data:
                anchor_key = 'anchor_text'
            
            if not anchor_key or not data[anchor_key]:
                # --- MODIFIED: Use progress_bar.write() for logging ---
                progress_bar.write(f"\n[Item {i}] SKIPPING: No 'anchor_story' or 'anchor_text' found.")
                save_state(CHECKPOINT_FILE, i) # Save state to skip this bad item
                continue
            
            original_anchor = data[anchor_key]
            new_items_list = []
            
            # --- MODIFIED: Add detailed logging ---
            progress_bar.write(f"\n[Item {i}/{len(input_data)-1}] Processing anchor: {original_anchor[:60]}...")
            
            # --- Generate and write augmented versions ---
            for lang in loaded_languages:
                log_prefix = f"  > [lang: {lang}]"
                progress_bar.write(f"{log_prefix} Translating en->{lang} (chunked)...")
                
                # Get the correct models for this language
                en_to_x_tok = models[f'en_to_{lang}']['tokenizer']
                en_to_x_mod = models[f'en_to_{lang}']['model']
                x_to_en_tok = models[f'{lang}_to_en']['tokenizer']
                x_to_en_mod = models[f'{lang}_to_en']['model']
                
                # --- UPDATED: Call translate_full_text ---
                intermediate_text = translate_full_text(
                    original_anchor, en_to_x_mod, en_to_x_tok, device, progress_bar, 
                    log_prefix=log_prefix
                )
                if not intermediate_text:
                    progress_bar.write(f"{log_prefix} FAILED (en->{lang}). Skipping.")
                    continue
                    
                progress_bar.write(f"{log_prefix} Translating {lang}->en (chunked)...")
                back_translated_anchor = translate_full_text(
                    intermediate_text, x_to_en_mod, x_to_en_tok, device, progress_bar,
                    log_prefix=log_prefix
                )
                # -------------------------------------------

                if not back_translated_anchor:
                    progress_bar.write(f"{log_prefix} FAILED ({lang}->en). Skipping.")
                    continue
                
                progress_bar.write(f"{log_prefix} Success. New text: {back_translated_anchor[:60]}...")
                
                # Create the new data object
                new_data = data.copy() # IMPORTANT: Copy the original data
                new_data[anchor_key] = back_translated_anchor
                new_data['augmentation_source'] = f"backtranslate_{lang}"
                new_items_list.append(new_data)
            
            # --- Checkpoint: Save all new items and update the state ---
            if new_items_list:
                for new_item in new_items_list:
                    append_jsonl(OUTPUT_FILE, new_item)
                
                save_state(CHECKPOINT_FILE, i)
                progress_bar.write(f"[Item {i}] SUCCESS: Saved {len(new_items_list)} new augmentations.")
                completed_count_session += 1
            else:
                progress_bar.write(f"[Item {i}] FAILED: No augmentations generated. Skipping.")
                # We still save the state to avoid retrying this item
                save_state(CHECKPOINT_FILE, i)
        
        print("\n--- Augmentation Script Finished ---")
        total_processed = start_index + completed_count_session
        print(f"Processed {completed_count_session} items in this session.")
        print(f"Total items processed: {total_processed}")
        print(f"Augmented data saved to: {OUTPUT_FILE}")

    except KeyboardInterrupt:
        print(f"\n\nProcess interrupted by user.")
        print(f"Progress saved. Last completed item was index: {load_state(CHECKPOINT_FILE)}")
        print(f"Run the script again to resume.")
    except Exception as e:
        print(f"\n\nUnexpected error: {e}")
        print(f"Progress saved. Last completed item was index: {load_state(CHECKPOINT_FILE)}")
        print("Run the script again to resume.")

if __name__ == "__main__":
    main()

