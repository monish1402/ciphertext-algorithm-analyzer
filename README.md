# CipherDetect - AI-Based Cryptographic Algorithm Analyzer

CipherDetect is an AI-powered cryptographic algorithm analyzer designed to identify encryption algorithms used in a given ciphertext. The system leverages machine learning models to classify and detect cryptographic methods, even in complex or obfuscated environments. This enables security analysts and researchers to respond swiftly to data breaches and analyze encrypted data efficiently.

## Features

- **AI-Powered Detection**: Utilizes deep learning models (XGBoost, MLP) to analyze and classify cryptographic algorithms.
- **Wide Algorithm Coverage**: Supports AES, DES, 3DES, RSA, Blowfish, SHA, and other cryptographic methods.
- **Feature Extraction**: Extracts statistical and frequency-based features (entropy, skewness, Fourier transforms, chi-square analysis, etc.).
- **Obfuscation Resilience**: Capable of identifying encryption methods even with partial or altered ciphertext.
- **Real-Time Analysis**: Fast processing and inference for immediate detection.
- **Model Training & Testing**: Supports dataset generation, hyperparameter tuning, and performance evaluation.
- **Docker Support**: Easily deployable using Docker and Docker Compose.

## Installation

### Prerequisites

Ensure you have the following installed:

- Python 3.8+
- pip
- Docker & Docker Compose

### Setup

Clone the repository and install dependencies:

```sh
 git clone https://github.com/monish1402/ciphertext-algorithm-analyzer.git
 cd ciphertext-algorithm-analyzer
 pip install -r requirements.txt
```

## Model Training

To train the model with your dataset:

```sh
python train_model.py
```

The best-performing model (XGBoost, MLP, or Random Forest) is saved as `best_crypto_model.pkl`.

## Testing

Run the test script to evaluate predictions:

```sh
python test_model.py
```

## Docker Deployment

### Using Docker Compose

To deploy the full stack (backend, frontend, and Nginx proxy):

```sh
docker-compose up --build
```

## Roadmap

- Improve model accuracy with adversarial training.
- Extend support for more encryption algorithms.
- Develop a browser extension for real-time web encryption detection.
- Deploy a cloud-based API for broader accessibility.

## Contributing

We welcome contributions! Feel free to submit issues, feature requests, or pull requests.

## License

This project is licensed under the MIT License.

---

### Maintainer

Developed by [Monishwaran](https://github.com/monish1402)

For inquiries, reach out via email or GitHub issues.

