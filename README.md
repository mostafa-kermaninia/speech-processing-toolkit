# Speaker Identification and Gender Classification

This repository contains the implementation of a Machine Learning pipeline for **Speaker Identification** and **Gender Classification** using audio features.

## 🚀 Project Overview

The goal of this project is to develop robust models that can:
1.  **Classify Gender**: Determine whether a speaker is male or female.
2.  **Identify Speakers**: Distinguish between different speakers based on their voice characteristics.

The project utilizes comprehensive audio signal processing techniques and state-of-the-art machine learning algorithms, ranging from classical classifiers (SVM, KNN, XGBoost) to Neural Networks.

## 📂 Repository Structure

```
├── data/                   # Data directory (raw and processed)
├── notebooks/              # Jupyter notebooks for experimentation
├── scripts/                # Executable scripts for training and evaluation
├── src/                    # Source code for the project
│   ├── data/               # Data loading and cleaning
│   ├── features/           # Audio processing and feature extraction
│   ├── models/             # Model definitions (Sklearn, Keras, etc.)
│   └── visualization/      # Plotting and evaluation utilities
├── requirements.txt        # Project dependencies
├── setup.py                # Package setup script
└── README.md               # Project documentation
```

## 🛠️ Installation

1.  **Clone the repository**:
    ```bash
    git clone https://github.com/your-username/Speaker-ID-Gender-Classification.git
    cd Speaker-ID-Gender-Classification
    ```

2.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    pip install -e .
    ```

## 📊 Methodology

### Feature Extraction
We extract a rich set of audio features including:
-   **Spectral Features**: MFCC, Spectral Centroid, Bandwidth, Contrast, Roll-off.
-   **Temporal Features**: Zero Crossing Rate, RMS Energy.
-   **Prosodic Features**: Fundamental Frequency (F0), Jitter, Shimmer.

### Processing Pipeline
1.  **Silence Removal**: Trimming silence using spectral centroid based windowing.
2.  **Noise Reduction**: Spectral subtraction to enhance signal quality.
3.  **Filtering**: Bandpass filter (80Hz - 5000Hz) to isolate human speech frequencies.
4.  **Resampling**: Standardizing sample rate to 44.1kHz.

### Models
We experiment with multiple architectures:
-   **Support Vector Machine (SVM)**: RBF kernel for non-linear separation.
-   **K-Nearest Neighbors (KNN)**: Baseline distance-based classifier.
-   **XGBoost / AdaBoost**: Ensemble methods for improved robustness.
-   **Multi-Layer Perceptron (MLP)**: Deep learning approach using Keras/TensorFlow.

## 🏃‍♂️ Usage

### 1. Download Data
The dataset is hosted on Google Drive. Run the setup script to download and structure the data:
```bash
python src/data/download.py
```

### 2. Train Gender Classifier
To train and evaluate the gender classification model:
```bash
python scripts/train_gender.py --model svm
```
*Available models: `svm`, `knn`, `xgboost`, `adaboost`.*

## 📈 Results

| Model | Accuracy | Precision | Recall |
|-------|----------|-----------|--------|
| SVM   | 0.98     | 0.98      | 0.98   |
| XGBoost| 0.97    | 0.97      | 0.97   |
| KNN   | 0.96     | 0.96      | 0.95   |

*(Note: Results may vary slightly based on random seed and data split)*

## 👥 Contributors

-   **Mostafa Kermani Nia** - Lead Developer & Researcher

## 📄 License

This project is licensed under the MIT License.
