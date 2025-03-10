from flask import Flask, request, jsonify
import joblib
from flask_cors import CORS  
import numpy as np
import pandas as pd
from scipy.stats import entropy, chisquare, skew, kurtosis  
from Cryptodome.Cipher import AES, DES, Blowfish, PKCS1_OAEP
from Cryptodome.PublicKey import RSA
from Cryptodome.Hash import SHA256
from Cryptodome.Random import get_random_bytes

app = Flask(__name__)
CORS(app)  

model = joblib.load("crypto_model.pkl")
print("✅ Loaded XGBoost Model!")

def pad(data, block_size):
    padding_length = block_size - (len(data) % block_size)
    return data + bytes([padding_length] * padding_length)

def shannon_entropy(ciphertext):
    _, counts = np.unique(ciphertext, return_counts=True)
    probs = counts / len(ciphertext)
    return -np.sum(probs * np.log2(probs + 1e-9))  

def extract_features(ciphertext):
    hist, _ = np.histogram(ciphertext, bins=256, range=(0, 256))
    prob = hist / np.sum(hist)  
    fft_vals = np.abs(np.fft.fft(ciphertext))[:30]  

    features = {
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

    return features

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Crypto Algorithm Identifier API is running!"})

@app.route("/predict", methods=["POST"])
def predict():
    try:
       
        data = request.get_json()

        if "ciphertext" not in data:
            return jsonify({"error": "Missing 'ciphertext' field in request."}), 400

        
        ciphertext = np.array(data["ciphertext"])

        if len(ciphertext) == 0:
            return jsonify({"error": "Ciphertext cannot be empty."}), 400

       
        features = extract_features(ciphertext)
        features_df = pd.DataFrame([features])

        
        prediction = model.predict(features_df)
        predicted_algorithm = ["AES", "DES", "RSA", "Blowfish", "SHA"][prediction[0]]

        return jsonify({"algorithm": predicted_algorithm})

    except Exception as e:
        return jsonify({"error": f"Error processing the input: {str(e)}"}), 500

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
