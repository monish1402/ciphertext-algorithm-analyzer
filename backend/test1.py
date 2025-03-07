import numpy as np
import joblib
import random
import string
from Cryptodome.Cipher import AES, DES, Blowfish, DES3
from Cryptodome.Hash import SHA256
from scipy.stats import entropy, chisquare, skew, kurtosis
import pandas as pd

# Load the trained model and scaler
model = joblib.load("best_crypto_model.pkl")
scaler = joblib.load("scaler.pkl")

# Algorithm mapping from training
algorithm_mapping = ["AES", "DES", "3DES", "Blowfish", "SHA", "Present", "Generalized"]

# 🔹 Generate Test Ciphertext
def generate_ciphertext(algorithm):
    key = bytes(range(16))
    plaintext = bytes(range(16))

    if algorithm == "AES":
        cipher = AES.new(key, AES.MODE_ECB)
        return np.frombuffer(cipher.encrypt(plaintext * 2), dtype=np.uint8)
    elif algorithm == "DES":
        cipher = DES.new(key[:8], DES.MODE_ECB)
        return np.frombuffer(cipher.encrypt(plaintext[:8] * 2), dtype=np.uint8)
    elif algorithm == "3DES":
        cipher = DES3.new(key[:24], DES3.MODE_ECB)
        return np.frombuffer(cipher.encrypt(plaintext[:8] * 2), dtype=np.uint8)
    elif algorithm == "Blowfish":
        cipher = Blowfish.new(key, Blowfish.MODE_ECB)
        return np.frombuffer(cipher.encrypt(plaintext * 2), dtype=np.uint8)
    elif algorithm == "SHA":
        hash_obj = SHA256.new(plaintext)
        return np.frombuffer(hash_obj.digest(), dtype=np.uint8)
    elif algorithm == "Present":
        return np.random.randint(0, 256, 64, dtype=np.uint8)
    elif algorithm == "Generalized":
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=32))
        return np.frombuffer(text.encode(), dtype=np.uint8)
    else:
        return np.random.randint(0, 256, 256, dtype=np.uint8)

# 🔹 Feature Extraction for Testing
def extract_features(ciphertext):
    hist, _ = np.histogram(ciphertext, bins=256, range=(0, 256))
    prob = hist / np.sum(hist)
    fft_vals = np.abs(np.fft.fft(ciphertext))[:50]
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
        "skewness": skew(ciphertext),
        "kurtosis": kurtosis(ciphertext)
    }

# 🔹 Test the Model with a New Ciphertext
actual_algorithm = random.choice(algorithm_mapping)  # Select a random algorithm
ciphertext = generate_ciphertext(actual_algorithm)  # Generate test ciphertext

# Extract features
features = extract_features(ciphertext)
features_df = pd.DataFrame([features])

# Scale the features using the saved scaler
features_scaled = scaler.transform(features_df)

# Predict the algorithm
predicted_label = model.predict(features_scaled)[0]
predicted_algorithm = algorithm_mapping[predicted_label]

# Print Results
print("\n✅ Loaded Best Crypto Model!")
print(f"\n🔍 Actual Algorithm: {actual_algorithm}")
print(f"🧠 Predicted Algorithm: {predicted_algorithm}")

# Check if prediction is correct
if actual_algorithm == predicted_algorithm:
    print("✅ Prediction is CORRECT! 🎯")
else:
    print("❌ Prediction is INCORRECT! 🚨")
