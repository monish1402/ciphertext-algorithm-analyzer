import numpy as np
import pandas as pd
import random
import joblib
import xgboost as xgb
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.stats import entropy, chisquare, skew, kurtosis
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from imblearn.over_sampling import SMOTE  
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, LeakyReLU
from tensorflow.keras.optimizers import Adam

algorithms = ["AES", "DES", "RSA", "Blowfish", "SHA"]

def generate_ciphertext(algorithm, length=128):
    if algorithm == "AES":
        return np.random.randint(0, 256, length)
    elif algorithm == "DES":
        return np.random.randint(0, 256, 64)
    elif algorithm == "RSA":
        return np.random.randint(0, 256, 256)
    elif algorithm == "Blowfish":
        return np.random.randint(0, 256, 64)
    elif algorithm == "SHA":
        return np.random.randint(0, 256, 32)
    else:
        return np.random.randint(0, 256, length)

def shannon_entropy(ciphertext):
    _, counts = np.unique(ciphertext, return_counts=True)
    probs = counts / len(ciphertext)
    return -np.sum(probs * np.log2(probs + 1e-9))  

def extract_features(ciphertext):
    hist, _ = np.histogram(ciphertext, bins=256, range=(0, 256))
    prob = hist / np.sum(hist)  
    fft_vals = np.abs(np.fft.fft(ciphertext))[:30]  

    return {
        "mean": np.mean(ciphertext),
        "std_dev": np.std(ciphertext),
        "entropy": entropy(prob + 1e-9),
        "shannon_entropy": shannon_entropy(ciphertext),
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
        "huffman_code_length": len(set(ciphertext)) 
    }

num_samples = 5000 
data = []
for _ in range(num_samples):
    algo = random.choice(algorithms)
    ciphertext = generate_ciphertext(algo)
    features = extract_features(ciphertext)
    features["algorithm"] = algo
    data.append(features)

df = pd.DataFrame(data)

df["algorithm"] = df["algorithm"].astype("category").cat.codes  

X = df.drop(columns=["algorithm"])
y = df["algorithm"]

smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(
    X_balanced, y_balanced, test_size=0.2, stratify=y_balanced, random_state=42
)

param_grid = {
    "n_estimators": [300, 500],
    "max_depth": [10, 15],
    "learning_rate": [0.05, 0.1],
    "subsample": [0.7, 0.9],  
    "colsample_bytree": [0.7, 0.9]
}

grid_search = GridSearchCV(
    xgb.XGBClassifier(eval_metric="mlogloss"), param_grid, cv=5, n_jobs=-1
)
grid_search.fit(X_train, y_train)

xgb_model = xgb.XGBClassifier(**grid_search.best_params_)
xgb_model.fit(X_train, y_train)

mlp_model = Sequential([
    Dense(128, input_shape=(X_train.shape[1],), activation=LeakyReLU(alpha=0.01)),
    Dropout(0.3),
    Dense(64, activation=LeakyReLU(alpha=0.01)),
    Dropout(0.3),
    Dense(len(set(y_train)), activation="softmax")  
])

mlp_model.compile(
    optimizer=Adam(learning_rate=0.001),  
    loss="sparse_categorical_crossentropy", 
    metrics=["accuracy"]
)

mlp_model.fit(X_train, y_train, epochs=100, batch_size=64, validation_data=(X_test, y_test))

y_pred_xgb = xgb_model.predict(X_test)
y_pred_mlp = np.argmax(mlp_model.predict(X_test), axis=1)

acc_xgb = accuracy_score(y_test, y_pred_xgb)
acc_mlp = accuracy_score(y_test, y_pred_mlp)

if acc_xgb > acc_mlp:
    joblib.dump(xgb_model, "crypto_model.pkl")  
    print(f"Saved XGBoost Model (Accuracy: {acc_xgb:.4f})")
else:
    mlp_model.save("crypto_model.h5")  
    print(f"Saved MLP Model (Accuracy: {acc_mlp:.4f})")

# Print evaluation metrics
print("\n🔹 Classification Report (XGBoost):\n", classification_report(y_test, y_pred_xgb, zero_division=1))
print("\n🔹 Classification Report (MLP Neural Network):\n", classification_report(y_test, y_pred_mlp, zero_division=1))

# Confusion Matrix for XGBoost
plt.figure(figsize=(8,6))
sns.heatmap(confusion_matrix(y_test, y_pred_xgb), annot=True, fmt="d", cmap="Blues", xticklabels=algorithms, yticklabels=algorithms)
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix - XGBoost Model")
plt.show()
