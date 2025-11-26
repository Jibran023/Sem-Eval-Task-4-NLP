import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.model_selection import StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from scipy.spatial.distance import cosine, euclidean, cityblock
import pickle
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 Train/Test Split - Clean Implementation (No Data Leakage)")
print("=" * 70)


def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data

print("\n🔹 Loading datasets...")

# Load main training set
sample_data = load_jsonl('data\\sample_track_a.jsonl')
print(f"✅ Sample data: {len(sample_data)} examples")

# Try loading synthetic data to expand training
try:
    synthetic_data = load_jsonl('data\\synthetic_data_for_classification.jsonl')
    print(f"✅ Synthetic data: {len(synthetic_data)} examples")
    synthetic_data = synthetic_data[:1900]   # use subset for faster runs
    train_data = sample_data + synthetic_data
except FileNotFoundError:
    print("⚠️ Synthetic data missing, continuing with sample only")
    train_data = sample_data

# Load dev set (unseen test)
test_data = load_jsonl('data\\dev_track_a.jsonl')
print(f"✅ Dev/test data: {len(test_data)} examples")

print(f"\n📊 Train: {len(train_data)} | Test: {len(test_data)}")



print("\n🔹 Loading Sentence Transformer (all-mpnet-base-v2)...")
model = SentenceTransformer('all-mpnet-base-v2')
print("✅ Transformer ready")



print("\n🔹 Encoding training data...")
train_anchor = [x['anchor_text'] for x in train_data]
train_a = [x['text_a'] for x in train_data]
train_b = [x['text_b'] for x in train_data]
train_labels = [1 if x['text_a_is_closer'] else 0 for x in train_data]

# Convert texts to embeddings
train_anchor_emb = model.encode(train_anchor, batch_size=32, show_progress_bar=True)
train_a_emb = model.encode(train_a, batch_size=32, show_progress_bar=True)
train_b_emb = model.encode(train_b, batch_size=32, show_progress_bar=True)
print("✅ Training embeddings created")



print("\n🔹 Encoding test data...")
test_anchor = [x['anchor_text'] for x in test_data]
test_a = [x['text_a'] for x in test_data]
test_b = [x['text_b'] for x in test_data]
test_labels = [1 if x['text_a_is_closer'] else 0 for x in test_data]

test_anchor_emb = model.encode(test_anchor, batch_size=32, show_progress_bar=True)
test_a_emb = model.encode(test_a, batch_size=32, show_progress_bar=True)
test_b_emb = model.encode(test_b, batch_size=32, show_progress_bar=True)
print("✅ Test embeddings created")



def create_enhanced_features(anchor, a, b):
    """Compute similarity/distance-based features between embeddings"""
    eps = 1e-10

    # Pairwise cosine similarities
    sim_a = 1 - cosine(anchor, a)
    sim_b = 1 - cosine(anchor, b)

    # Euclidean + Manhattan distances
    dist_a = euclidean(anchor, a)
    dist_b = euclidean(anchor, b)
    man_a = cityblock(anchor, a)
    man_b = cityblock(anchor, b)

    # Dot products
    dot_a = np.dot(anchor, a)
    dot_b = np.dot(anchor, b)

    # Cross similarities (A vs B)
    sim_ab = 1 - cosine(a, b)
    dist_ab = euclidean(a, b)

    # Difference + ratio features
    sim_diff = sim_a - sim_b
    dist_diff = dist_b - dist_a
    dot_diff = dot_a - dot_b
    sim_ratio = sim_a / (sim_b + eps)
    dist_ratio = dist_b / (dist_a + eps)
    dot_ratio = dot_a / (dot_b + eps)

    # Extra derived features
    angle_a = np.arccos(np.clip(sim_a, -1, 1))
    angle_b = np.arccos(np.clip(sim_b, -1, 1))
    angle_diff = angle_a - angle_b

    # Basic stats of embeddings
    mean_anchor, mean_a, mean_b = np.mean(anchor), np.mean(a), np.mean(b)
    std_anchor, std_a, std_b = np.std(anchor), np.std(a), np.std(b)

    # Norms
    norm_anchor, norm_a, norm_b = np.linalg.norm(anchor), np.linalg.norm(a), np.linalg.norm(b)

    # Interaction features
    sim_product = sim_a * sim_b
    harmonic_mean = 2 * (sim_a * sim_b) / (sim_a + sim_b + eps)
    max_sim, min_sim = max(sim_a, sim_b), min(sim_a, sim_b)

    # Derived metrics
    confidence = abs(sim_diff) / (max(sim_a, sim_b) + eps)
    relative_sim = (sim_a - sim_b) / (sim_a + sim_b + eps)
    log_sim_ratio = np.log(sim_ratio + eps)
    log_dist_ratio = np.log(dist_ratio + eps)

    # Cross combinations
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

print("\n🔹 Building train/test feature matrices...")

# Vectorize train
X_train = np.array([
    create_enhanced_features(a, b, c)
    for a, b, c in zip(train_anchor_emb, train_a_emb, train_b_emb)
])
y_train = np.array(train_labels)

# Vectorize test
X_test = np.array([
    create_enhanced_features(a, b, c)
    for a, b, c in zip(test_anchor_emb, test_a_emb, test_b_emb)
])
y_test = np.array(test_labels)

print(f"✅ Train shape: {X_train.shape}, Test shape: {X_test.shape}")



print("\n" + "=" * 70)
print("🧠 Phase 1: Cross-Validation (Tuning MLP Layers)")
print("=" * 70)

# Normalize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)

# Different network depths to test
mlp_configs = [
    {'name': 'Small', 'hidden': (64, 32), 'alpha': 0.001},
    {'name': 'Medium', 'hidden': (128, 64, 32), 'alpha': 0.001},
    {'name': 'Large', 'hidden': (150, 75), 'alpha': 0.0001},
    {'name': 'Deep', 'hidden': (256, 128, 64), 'alpha': 0.0001},
]

best_config = None
best_cv = 0

# 5-fold CV for each config
for cfg in mlp_configs:
    print(f"\n➡️  Testing: {cfg['name']} | Layers: {cfg['hidden']}")
    mlp = MLPClassifier(
        hidden_layer_sizes=cfg['hidden'],
        alpha=cfg['alpha'],
        activation='relu',
        solver='adam',
        learning_rate_init=0.001,
        batch_size=32,
        max_iter=2000,
        random_state=42,
        early_stopping=True
    )

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []

    for train_idx, val_idx in skf.split(X_train_scaled, y_train):
        X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[train_idx], y_train[val_idx]
        mlp.fit(X_tr, y_tr)
        scores.append(mlp.score(X_val, y_val))

    mean_cv = np.mean(scores)
    print(f"   Avg CV: {mean_cv*100:.2f}% ± {np.std(scores)*100:.2f}%")

    if mean_cv > best_cv:
        best_cv = mean_cv
        best_config = cfg

print(f"\n🏆 Best config: {best_config['name']} | CV: {best_cv*100:.2f}%")



print("\n" + "=" * 70)
print("🧠 Phase 2: Final Training with Best Config")
print("=" * 70)

final_mlp = MLPClassifier(
    hidden_layer_sizes=best_config['hidden'],
    alpha=best_config['alpha'],
    activation='relu',
    solver='adam',
    learning_rate_init=0.001,
    batch_size=32,
    max_iter=2000,
    random_state=42,
    early_stopping=True
)

print("🔹 Training final model...")
final_mlp.fit(X_train_scaled, y_train)
print("✅ Final MLP trained successfully")



print("🎯 Phase 3: Evaluation on Held-out Test (dev_tracka)")


X_test_scaled = scaler.transform(X_test)
y_pred = final_mlp.predict(X_test_scaled)
acc = accuracy_score(y_test, y_pred)

print(f"\n🎯 Test Accuracy: {acc*100:.2f}% (real unseen data)")
print("\n📊 Classification Report:")
print(classification_report(y_test, y_pred, target_names=['Text B closer', 'Text A closer'], digits=4))

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("\n📊 Confusion Matrix:")
print(f"                    Pred B    Pred A")
print(f"  Actual B (0):     {cm[0][0]:6d}      {cm[0][1]:6d}")
print(f"  Actual A (1):     {cm[1][0]:6d}      {cm[1][1]:6d}")




print("📈 Comparison: Cross-Validation vs Test Performance")

print(f"   CV Mean: {best_cv*100:.2f}%")
print(f"   Test Acc: {acc*100:.2f}%")
print(f"   Gap: {(acc - best_cv)*100:+.2f}%")

if acc < best_cv - 0.05:
    print("⚠️  Possible overfitting — consider more data or regularization")
elif acc > best_cv:
    print("✅ Generalization looks solid")



print("\n💾 Saving model and scaler...")
with open('mlp_model_no_leakage.pkl', 'wb') as f:
    pickle.dump(final_mlp, f)
with open('scaler_no_leakage.pkl', 'wb') as f:
    pickle.dump(scaler, f)
print("✅ Saved → mlp_model_no_leakage.pkl / scaler_no_leakage.pkl")


print("✅ Training + Evaluation Complete (No Data Leakage)")
