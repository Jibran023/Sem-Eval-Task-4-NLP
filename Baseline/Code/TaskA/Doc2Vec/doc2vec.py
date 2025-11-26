import json
import numpy as np
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from scipy.spatial.distance import cosine, euclidean, cityblock
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("🚀 Doc2Vec for Track A - Optimized Implementation")
print("=" * 70)

def load_jsonl(filepath):
    data = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line.strip()))
    return data


print("\n🔹 Loading data...")

# Load core sample training data
sample_data = load_jsonl('data\\sample_track_a.jsonl')
print(f"   ✅ Sample data: {len(sample_data)} examples")

# Try adding synthetic data (optional, improves model variety)
try:
    synthetic_data = load_jsonl('data\\synthetic_data_for_classification.jsonl')
    print(f"   ✅ Synthetic data: {len(synthetic_data)} examples")

    # Use subset to control runtime
    synthetic_data = synthetic_data[:1900]
    print(f"   ℹ️  Using {len(synthetic_data)} synthetic examples")

    train_data = sample_data + synthetic_data
except FileNotFoundError:
    print("   ⚠️  Synthetic data not found, using only sample data")
    train_data = sample_data

# Load dev set (kept unseen)
test_data = load_jsonl('data\\dev_track_a.jsonl')
print(f"   ✅ Test data: {len(test_data)} examples")

print(f"\n📊 Data Split: Train={len(train_data)} | Test={len(test_data)}")



def preprocess_text(text):
    # Lowercase + simple token split
    if not isinstance(text, str):
        return []
    return text.lower().split()


print("\n🔹 Tagging and preparing documents...")

all_documents = []       # all TaggedDocument objects
doc_id = 0
train_doc_ids, test_doc_ids = [], []

# --- Training documents
for item in train_data:
    anchor_id = f'train_anchor_{doc_id}'
    text_a_id = f'train_a_{doc_id}'
    text_b_id = f'train_b_{doc_id}'

    # Add anchor, text_a, text_b as individual tagged docs
    all_documents.append(TaggedDocument(preprocess_text(item['anchor_text']), [anchor_id]))
    all_documents.append(TaggedDocument(preprocess_text(item['text_a']), [text_a_id]))
    all_documents.append(TaggedDocument(preprocess_text(item['text_b']), [text_b_id]))

    # Store mapping and label
    train_doc_ids.append({
        'anchor': anchor_id,
        'text_a': text_a_id,
        'text_b': text_b_id,
        'label': 1 if item['text_a_is_closer'] else 0
    })
    doc_id += 1

# --- Test documents
for item in test_data:
    anchor_id = f'test_anchor_{doc_id}'
    text_a_id = f'test_a_{doc_id}'
    text_b_id = f'test_b_{doc_id}'

    all_documents.append(TaggedDocument(preprocess_text(item['anchor_text']), [anchor_id]))
    all_documents.append(TaggedDocument(preprocess_text(item['text_a']), [text_a_id]))
    all_documents.append(TaggedDocument(preprocess_text(item['text_b']), [text_b_id]))

    test_doc_ids.append({
        'anchor': anchor_id,
        'text_a': text_a_id,
        'text_b': text_b_id,
        'label': 1 if item['text_a_is_closer'] else 0
    })
    doc_id += 1

print(f"✅ Prepared {len(all_documents)} tagged docs for training")


print("🧠 Training Doc2Vec model...")

# dm=1 → Distributed Memory, better for capturing context
doc2vec_model = Doc2Vec(
    documents=all_documents,
    vector_size=450,      # embedding size
    window=10,            # context window
    min_count=2,          # min freq for word
    workers=4,            # threads
    epochs=100,           # training epochs
    dm=1,                 # distributed memory
    dm_mean=1,            # use mean context
    alpha=0.025,
    min_alpha=0.0001,
    seed=42
)

print(f"\n✅ Model trained | Vocab size: {len(doc2vec_model.wv)} | Vec dim: {doc2vec_model.vector_size}")

def get_vector(doc_id):
    return doc2vec_model.dv[doc_id]

# Build vector arrays for train/test
train_vectors = {'anchors': [], 'text_a': [], 'text_b': [], 'labels': []}
for item in train_doc_ids:
    train_vectors['anchors'].append(get_vector(item['anchor']))
    train_vectors['text_a'].append(get_vector(item['text_a']))
    train_vectors['text_b'].append(get_vector(item['text_b']))
    train_vectors['labels'].append(item['label'])

test_vectors = {'anchors': [], 'text_a': [], 'text_b': [], 'labels': []}
for item in test_doc_ids:
    test_vectors['anchors'].append(get_vector(item['anchor']))
    test_vectors['text_a'].append(get_vector(item['text_a']))
    test_vectors['text_b'].append(get_vector(item['text_b']))
    test_vectors['labels'].append(item['label'])

print("✅ Extracted all document vectors")


print("\n🔹 Generating feature vectors...")

def create_doc2vec_features(anchor_vec, text_a_vec, text_b_vec):
    # Clean any NaN/infinite values
    anchor_vec = np.nan_to_num(anchor_vec)
    text_a_vec = np.nan_to_num(text_a_vec)
    text_b_vec = np.nan_to_num(text_b_vec)

    # Similarity and distance metrics
    sim_a_cos = 1 - cosine(anchor_vec, text_a_vec)
    sim_b_cos = 1 - cosine(anchor_vec, text_b_vec)
    dist_a_euc = euclidean(anchor_vec, text_a_vec)
    dist_b_euc = euclidean(anchor_vec, text_b_vec)
    dist_a_man = cityblock(anchor_vec, text_a_vec)
    dist_b_man = cityblock(anchor_vec, text_b_vec)
    dot_a = np.dot(anchor_vec, text_a_vec)
    dot_b = np.dot(anchor_vec, text_b_vec)

    # Differences between pairs
    sim_diff = sim_a_cos - sim_b_cos
    dist_diff = dist_b_euc - dist_a_euc
    dot_diff = dot_a - dot_b

    # Combine into one feature list
    return np.nan_to_num([
        sim_a_cos, sim_b_cos,
        dist_a_euc, dist_b_euc,
        dist_a_man, dist_b_man,
        dot_a, dot_b,
        sim_diff, dist_diff, dot_diff
    ])

# Create training features (anchor vs A/B)
X_train = np.array([
    create_doc2vec_features(a, b, c)
    for a, b, c in zip(train_vectors['anchors'], train_vectors['text_a'], train_vectors['text_b'])
])
y_train = np.array(train_vectors['labels'])

# Create testing features
X_test = np.array([
    create_doc2vec_features(a, b, c)
    for a, b, c in zip(test_vectors['anchors'], test_vectors['text_a'], test_vectors['text_b'])
])
y_test = np.array(test_vectors['labels'])

print(f"✅ Feature shape: {X_train.shape} | Test: {X_test.shape}")


print("🧩 Training classifiers...")

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define a few models
classifiers = {
    'Logistic Regression': LogisticRegression(max_iter=1000, C=1.0, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=200, learning_rate=0.05, max_depth=5, random_state=42),
    'MLP Neural Network': MLPClassifier(hidden_layer_sizes=(128, 64, 32), max_iter=2000, random_state=42, early_stopping=True)
}

# Try adding XGBoost if installed
try:
    from xgboost import XGBClassifier
    classifiers['XGBoost'] = XGBClassifier(n_estimators=200, max_depth=5, learning_rate=0.05, random_state=42)
except ImportError:
    pass

results = {}

# Loop over each classifier
for name, clf in classifiers.items():
    print(f"\n{'='*70}")
    print(f"📊 {name}")

    # Run quick 5-fold CV
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(skf.split(X_train_scaled, y_train), 1):
        X_tr, X_val = X_train_scaled[tr_idx], X_train_scaled[val_idx]
        y_tr, y_val = y_train[tr_idx], y_train[val_idx]
        model = type(clf)(**clf.get_params())
        model.fit(X_tr, y_tr)
        score = model.score(X_val, y_val)
        scores.append(score)
        print(f"   Fold {fold}: {score*100:.2f}%")

    mean_cv = np.mean(scores)
    std_cv = np.std(scores)

    # Retrain full model
    clf.fit(X_train_scaled, y_train)
    y_pred = clf.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)

    print(f"   Test Accuracy: {acc*100:.2f}%")
    results[name] = {'cv_mean': mean_cv, 'cv_std': std_cv, 'test_acc': acc}

print("📈 Final Results Summary")

print(f"\n{'Model':<25} {'CV Mean':<15} {'Test Acc':<10}")

for name, r in sorted(results.items(), key=lambda x: x[1]['test_acc'], reverse=True):
    print(f"{name:<25} {r['cv_mean']*100:.2f}% ±{r['cv_std']*100:.2f}%   {r['test_acc']*100:.2f}%")

best_model_name = max(results, key=lambda x: results[x]['test_acc'])
print(f"\n🏆 Best Model: {best_model_name} ({results[best_model_name]['test_acc']*100:.2f}%)")

import pickle
print("\n💾 Saving models...")

# Saveing Doc2Vec embeddings + best classifier + scaler
doc2vec_model.save('Work\\TaskA\\Others\\Doc2Vec\\doc2vec_model.bin')
with open('Work\\TaskA\\Others\\Doc2Vec\\doc2vec_best_classifier.pkl', 'wb') as f:
    pickle.dump(classifiers[best_model_name], f)
with open('Work\\TaskA\\Others\\Doc2Vec\\doc2vec_scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

print("✅ Models saved successfully.")
