import numpy as np
import pandas as pd
import random
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import RobustScaler
from Cryptodome.Cipher import AES, DES, Blowfish, PKCS1_OAEP
from Cryptodome.PublicKey import RSA
from Cryptodome.Hash import SHA256
from Cryptodome.Random import get_random_bytes
from sklearn.utils.class_weight import compute_class_weight

# 🔹 Step 1: Define Cryptographic Algorithms
algorithms = ["AES", "DES", "RSA", "Blowfish", "SHA"]

# 🔹 Step 2: Generate Randomized Plaintext
def generate_plaintext():
    length = random.choice([16, 32, 64, 128, 256])
    return get_random_bytes(length)

# 🔹 Step 3: Generate Ciphertext
def generate_ciphertext(algorithm):
    plaintext = generate_plaintext()
    
    if algorithm == "AES":
        key = get_random_bytes(16)
        cipher = AES.new(key, AES.MODE_ECB)
        ciphertext = cipher.encrypt(plaintext.ljust(16))

    elif algorithm == "DES":
        key = get_random_bytes(8)
        cipher = DES.new(key, DES.MODE_ECB)
        ciphertext = cipher.encrypt(plaintext.ljust(8))

    elif algorithm == "RSA":
        key = RSA.generate(2048)
        cipher = PKCS1_OAEP.new(key)
        ciphertext = cipher.encrypt(plaintext[:190])  # Ensure it fits RSA encryption block
        ciphertext = np.frombuffer(ciphertext, dtype=np.uint8)

    elif algorithm == "Blowfish":
        key = get_random_bytes(16)
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        ciphertext = cipher.encrypt(plaintext.ljust(8))

    elif algorithm == "SHA":
        hash_obj = SHA256.new(plaintext)
        ciphertext = hash_obj.digest()

    else:
        return None

    return np.frombuffer(ciphertext, dtype=np.uint8)

# 🔹 Step 4: Feature Extraction
def extract_features(ciphertext):
    hist, _ = np.histogram(ciphertext, bins=256, range=(0, 256))
    prob = hist / np.sum(hist)
    fft_vals = np.abs(np.fft.fft(ciphertext))[:30]

    return {
        "mean": np.mean(ciphertext),
        "std_dev": np.std(ciphertext),
        "entropy": entropy(prob + 1e-9),
        "chi_square": chisquare(hist + 1)[0],
        "byte_variance": np.var(ciphertext),
        "median": np.median(ciphertext),
        "range": np.max(ciphertext) - np.min(ciphertext),
        "fourier_mean": np.mean(fft_vals),
        "fourier_std": np.std(fft_vals),
        "max_byte_freq": np.max(hist),
        "min_byte_freq": np.min(hist),
        "skewness": skew(ciphertext),
        "kurtosis": kurtosis(ciphertext),
        "huffman_code_length": len(set(ciphertext)),  # Huffman complexity
        "iqr": np.percentile(ciphertext, 75) - np.percentile(ciphertext, 25),  # Interquartile Range
        "mode_freq": np.max(hist) / np.sum(hist),  # Mode frequency
    }

# 🔹 Step 5: Generate Dataset
num_samples = 10000  # Further increased dataset size
data = []
for _ in range(num_samples):
    algo = random.choice(algorithms)
    ciphertext = generate_ciphertext(algo)
    features = extract_features(ciphertext)
    features["algorithm"] = algo
    data.append(features)

df = pd.DataFrame(data)

# 🔹 Step 6: Encode Labels
df["algorithm"] = df["algorithm"].astype("category").cat.codes

# 🔹 Step 7: Normalize Features with RobustScaler
scaler = RobustScaler()
X = scaler.fit_transform(df.drop(columns=["algorithm"]))
y = df["algorithm"]
joblib.dump(scaler, "scaler.pkl")  # Save scaler for testing consistency

# 🔹 Step 8: Handle Imbalanced Data with SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# 🔹 Step 9: Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

# 🔹 Step 10: Compute Class Weights
class_weights = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
weights_dict = {i: w for i, w in enumerate(class_weights)}

# 🔹 Step 11: Initialize Models
models = {
    "Random Forest": RandomForestClassifier(n_estimators=500, max_depth=30, class_weight="balanced"),
    "XGBoost": xgb.XGBClassifier(n_estimators=500, max_depth=20, learning_rate=0.03, scale_pos_weight=weights_dict),
    "MLP": MLPClassifier(hidden_layer_sizes=(256, 128, 64), max_iter=1000, activation='relu', solver='adam', random_state=42)
}

# 🔹 Step 12: Train and Evaluate Models
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average='macro', zero_division=1)
    rec = recall_score(y_test, y_pred, average='macro', zero_division=1)
    f1 = f1_score(y_test, y_pred, average='macro', zero_division=1)
    results[name] = {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1_score": f1,
        "confusion_matrix": confusion_matrix(y_test, y_pred)
    }
    print(f"\n🔹 {name} - Accuracy: {acc:.4f}, Precision: {prec:.4f}, Recall: {rec:.4f}, F1 Score: {f1:.4f}")
    sns.heatmap(confusion_matrix(y_test, y_pred), annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.show()

# 🔹 Step 13: Save the Best Model
best_model = max(results, key=lambda x: results[x]["accuracy"])
joblib.dump(models[best_model], "best_crypto_model.pkl")
print(f"\n🚀 Best Performing Model: {best_model} - Model Saved!")
