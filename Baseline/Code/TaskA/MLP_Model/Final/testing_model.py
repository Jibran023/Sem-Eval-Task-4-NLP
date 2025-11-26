import json
import numpy as np
import pickle
from sentence_transformers import SentenceTransformer
from scipy.spatial.distance import cosine, euclidean, cityblock
from sklearn.metrics import accuracy_score, classification_report


print("🧪 Evaluate MLP Model on Dev Track A")



# 1. Load saved MLP model and scaler

print("\n📂 Step 1: Loading trained model...")

try:
    # Load trained MLP model
    with open('Work\\TaskA\\Others\\Final\\mlp_model_no_leakage.pkl', 'rb') as f:
        mlp_model = pickle.load(f)

    # Load corresponding feature scaler
    with open('Work\\TaskA\\Others\\Final\\scaler_no_leakage.pkl', 'rb') as f:
        scaler = pickle.load(f)

    print("✅ Model and scaler loaded successfully")

except FileNotFoundError:
    print("❌ ERROR: Model files not found.")
    print("   Make sure you've run the training script first.")
    exit()



# 2. Load Dev Track A data

print("\n📂 Step 2: Loading dev_track_a.jsonl...")

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

try:
    test_data = load_jsonl('data\\dev_track_a.jsonl')
    print(f"✅ Loaded {len(test_data)} dev examples")
except FileNotFoundError:
    print("❌ ERROR: dev_track_a.jsonl missing")
    exit()



# 3. Load SBERT and encode texts

print("\n🤖 Step 3: Loading SBERT and encoding text samples...")

sbert = SentenceTransformer('all-mpnet-base-v2')

# Separate all fields from JSONL
anchor_texts = [x['anchor_text'] for x in test_data]
text_a_list = [x['text_a'] for x in test_data]
text_b_list = [x['text_b'] for x in test_data]
true_labels = [1 if x['text_a_is_closer'] else 0 for x in test_data]

# Encode all three sets of sentences
print("   Encoding anchors...")
anchor_embs = sbert.encode(anchor_texts, show_progress_bar=True, batch_size=32)

print("   Encoding text A...")
text_a_embs = sbert.encode(text_a_list, show_progress_bar=True, batch_size=32)

print("   Encoding text B...")
text_b_embs = sbert.encode(text_b_list, show_progress_bar=True, batch_size=32)

print("✅ Encodings complete")



# 4. Create same features as during training

print("\n🔧 Step 4: Building feature matrix...")

def create_features(anchor_emb, text_a_emb, text_b_emb):
    """Generate same feature vector as in training."""
    eps = 1e-10

    # Core similarities/distances
    sim_a = 1 - cosine(anchor_emb, text_a_emb)
    sim_b = 1 - cosine(anchor_emb, text_b_emb)
    dist_a = euclidean(anchor_emb, text_a_emb)
    dist_b = euclidean(anchor_emb, text_b_emb)
    man_a = cityblock(anchor_emb, text_a_emb)
    man_b = cityblock(anchor_emb, text_b_emb)
    dot_a = np.dot(anchor_emb, text_a_emb)
    dot_b = np.dot(anchor_emb, text_b_emb)

    # Cross-comparisons
    sim_ab = 1 - cosine(text_a_emb, text_b_emb)
    dist_ab = euclidean(text_a_emb, text_b_emb)

    # Diffs and ratios
    sim_diff = sim_a - sim_b
    dist_diff = dist_b - dist_a
    dot_diff = dot_a - dot_b
    sim_ratio = sim_a / (sim_b + eps)
    dist_ratio = dist_b / (dist_a + eps)
    dot_ratio = dot_a / (dot_b + eps)

    # Angular measures
    angle_a = np.arccos(np.clip(sim_a, -1, 1))
    angle_b = np.arccos(np.clip(sim_b, -1, 1))
    angle_diff = angle_a - angle_b

    # Mean/std of embeddings
    mean_anchor, mean_a, mean_b = np.mean(anchor_emb), np.mean(text_a_emb), np.mean(text_b_emb)
    std_anchor, std_a, std_b = np.std(anchor_emb), np.std(text_a_emb), np.std(text_b_emb)

    # Norms
    norm_anchor, norm_a, norm_b = np.linalg.norm(anchor_emb), np.linalg.norm(text_a_emb), np.linalg.norm(text_b_emb)

    # Combined features
    sim_product = sim_a * sim_b
    harmonic_mean = 2 * (sim_a * sim_b) / (sim_a + sim_b + eps)
    max_sim, min_sim = max(sim_a, sim_b), min(sim_a, sim_b)

    # Advanced metrics
    confidence = abs(sim_diff) / (max(sim_a, sim_b) + eps)
    relative_sim = (sim_a - sim_b) / (sim_a + sim_b + eps)
    log_sim_ratio = np.log(sim_ratio + eps)
    log_dist_ratio = np.log(dist_ratio + eps)

    # Cross and margin
    cross_1 = sim_a * dist_b
    cross_2 = sim_b * dist_a
    margin = abs(sim_a - sim_b)

    return np.array([
        sim_a, sim_b, dist_a, dist_b, man_a, man_b,
        dot_a, dot_b, sim_ab, dist_ab,
        sim_diff, dist_diff, dot_diff,
        sim_ratio, dist_ratio, dot_ratio,
        angle_a, angle_b, angle_diff,
        mean_anchor, mean_a, mean_b,
        std_anchor, std_a, std_b,
        norm_anchor, norm_a, norm_b,
        sim_product, harmonic_mean, max_sim, min_sim,
        confidence, relative_sim, log_sim_ratio, log_dist_ratio,
        cross_1, cross_2, margin
    ])

# Build feature vectors for all test rows
X_test = np.array([
    create_features(anchor_embs[i], text_a_embs[i], text_b_embs[i])
    for i in range(len(test_data))
])
print(f"✅ Feature matrix ready: {X_test.shape}")



# 5. Scale features using training scaler

print("\n⚖️ Step 5: Scaling features...")
X_test_scaled = scaler.transform(X_test)
print("✅ Scaled successfully")



# 6. Predict on test set

print("\n🎯 Step 6: Generating predictions...")
preds = mlp_model.predict(X_test_scaled)
probs = mlp_model.predict_proba(X_test_scaled)
print("✅ Predictions complete")



# 7. Evaluate results

print("📊 EVALUATION RESULTS")


acc = accuracy_score(true_labels, preds)
print(f"\n🎯 Accuracy: {acc * 100:.2f}%")

# Quick count summary
correct = np.sum(preds == true_labels)
incorrect = len(preds) - correct
print(f"   ✅ Correct:   {correct}/{len(preds)} ({correct/len(preds)*100:.1f}%)")
print(f"   ❌ Incorrect: {incorrect}/{len(preds)} ({incorrect/len(preds)*100:.1f}%)")

# Detailed classification breakdown
print("\n📋 Classification Report:")
print(classification_report(true_labels, preds,
      target_names=['Text B is closer', 'Text A is closer'], digits=2))



# 8. Example outputs (qualitative check)
print("🔍 SAMPLE PREDICTIONS")


# Display some correct predictions
print("\n✅ Correct examples:")
correct_idx = np.where(preds == true_labels)[0][:3]
for idx in correct_idx:
    pred_label = 'A' if preds[idx] == 1 else 'B'
    conf = np.max(probs[idx]) * 100
    print(f"\n   Example {idx}: Pred → Text {pred_label} ({conf:.1f}% conf) ✓")

# Display some incorrect ones
print("\n❌ Incorrect examples:")
wrong_idx = np.where(preds != true_labels)[0][:3]
for idx in wrong_idx:
    pred_label = 'A' if preds[idx] == 1 else 'B'
    true_label = 'A' if true_labels[idx] == 1 else 'B'
    conf = np.max(probs[idx]) * 100
    print(f"\n   Example {idx}: Pred → Text {pred_label} ({conf:.1f}% conf) ✗ Actual: {true_label}")

print("✅ Evaluation complete.")

