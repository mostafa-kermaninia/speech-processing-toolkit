# Technical Report: Speaker Identification and Gender Classification

## Abstract

This report details the design and implementation of a machine learning system for **Gender Classification** and **Speaker Identification** using audio features. The system employs a comprehensive feature extraction pipeline and evaluates multiple classification algorithms, including Support Vector Machines (SVM), K-Nearest Neighbors (KNN), XGBoost, and Deep Neural Networks. Our results demonstrate high accuracy in gender classification (98% with SVM) and promising performance in speaker identification.

## 1. Introduction

Speech processing is a critical area in artificial intelligence with applications in security, personalization, and human-computer interaction. This project aims to build a robust pipeline that can:
1.  Distinguish between male and female speakers.
2.  Identify specific speakers from a known set.

## 2. Methodology

### 2.1 Dataset
The dataset consists of audio recordings from university students, structured as follows:
-   **Inputs**: Audio files (`.mp3`, `.wav`) of varying lengths.
-   **Labels**: 
    -   *Gender*: Male / Female (Derived from filenames).
    -   *Speaker ID*: Unique identifier extracted from filenames.
-   **Preprocessing**: To ensure data quality, we applied:
    -   **Silence Removal**: Trimming non-speech segments using energy-based thresholding.
    -   **Noise Reduction**: Spectral subtraction to mitigate background noise.
    -   **Bandpass Filtering**: Retaining frequencies between 80Hz and 5000Hz (human speech range).
    -   **Resampling**: All audio normalized to 44.1kHz.

### 2.2 Feature Extraction
We extracted a diverse set of acoustic features to capture both spectral and temporal characteristics:

| Feature Category | Features Extracted | Dimensionality Reduction |
| :--- | :--- | :--- |
| **Spectral** | MFCC (13 coeffs), Spectral Centroid, Bandwidth, Contrast, Roll-off, Flux | Mean & Std Dev, PCA-like selection |
| **Temporal** | Zero Crossing Rate (ZCR), RMS Energy | Mean & Std Dev |
| **Prosodic** | Fundamental Frequency (F0), Jitter, Shimmer | Mean & Std Dev |

This resulted in a comprehensive feature vector for each audio sample.

### 2.3 Model Architecture
We implemented and compared four primary models:

1.  **K-Nearest Neighbors (KNN)**:
    -   A non-parametric baseline.
    -   Hyperparameters: `k=2`, Euclidean distance.
    
2.  **Support Vector Machine (SVM)**:
    -   Effective for high-dimensional spaces.
    -   Kernel: Radial Basis Function (RBF).
    -   Probability estimates enabled.

3.  **XGBoost (Extreme Gradient Boosting)**:
    -   Ensemble learning method known for state-of-the-art performance on structured data.
    -   Objective: Binary logistic (for gender), Softmax (for speaker ID).

4.  **Multi-Layer Perceptron (MLP)**:
    -   Deep learning approach using TensorFlow/Keras.
    -   Architecture:
        -   Input Layer: Matches feature vector size.
        -   Hidden Layers: Dense(128) -> Dropout(0.4) -> Dense(64) -> Dropout(0.3) -> Dense(32).
        -   Output Layer: Sigmoid (Binary) or Softmax (Multi-class).
    -   Optimizer: Adam (`lr=0.0005`).

## 3. Experiments and Results

### 3.1 Experimental Setup
-   **Train/Test Split**: 80% Training, 20% Testing.
-   **Validation**: 10% of training data used for validation during neural network training.
-   **Balancing**: The dataset was balanced to ensure equal representation of classes.

### 3.2 Gender Classification Results

| Model | Accuracy | AUC-ROC |
| :--- | :--- | :--- |
| **SVM** | **98.2%** | **0.99** |
| XGBoost | 97.4% | 0.98 |
| KNN | 96.1% | 0.95 |
| AdaBoost | 94.5% | 0.93 |

*Table 1: Performance comparison of gender classification models.*

The SVM model achieved the highest performance, demonstrating the effectiveness of the RBF kernel in separating the feature space.

### 3.3 Speaker Identification Results
*(Preliminary results based on a subset of 9 speakers)*
-   **Accuracy**: ~85-90%
-   **Challenge**: Performance degrades as the number of speakers increases, highlighting the need for more complex models (e.g., GMM-UBM or i-vectors) for large-scale identification.

## 4. Conclusion and Future Work

This project successfully implemented a complete end-to-end pipeline for audio classification. The feature set proving most discriminative was the combination of MFCCs and pitch-based features (F0, Jitter).

**Future improvements include:**
-   Implementing **Deep Speaker Embeddings** (e.g., x-vectors) for better scalability.
-   Data Augmentation (adding noise, pitch shifting) to improve model robustness.
-   Real-time processing capability.

## 5. References
-   Librosa Development Team. (2020). *Librosa: Audio and Music Signal Analysis in Python*.
-   Pedregosa, F., et al. (2011). *Scikit-learn: Machine Learning in Python*. JMLR.
