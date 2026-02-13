import os
import argparse
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report

from src.config import RAW_DATA_DIR, PROCESSED_DATA_DIR, RANDOM_SEED, TEST_SIZE
from src.data.dataset import get_audio_files, categorize_files_by_gender, create_balanced_dataset
from src.features.audio_processing import preprocess_audio_file
from src.features.feature_extraction import extract_features
from src.models.classifiers import get_knn_model, get_svm_model, get_xgboost_model, get_adaboost_model
from src.visualization.plots import plot_confusion_matrix, plot_roc_curve

def main():
    parser = argparse.ArgumentParser(description="Train Gender Classification Models")
    parser.add_argument("--model", type=str, default="svm", choices=["knn", "svm", "xgboost", "adaboost"], help="Model to train")
    parser.add_argument("--data_dir", type=str, default=os.path.join(RAW_DATA_DIR, "HW1_M"), help="Path to audio files")
    args = parser.parse_args()

    # 1. Load Data
    print("Loading data...")
    audio_files = get_audio_files(args.data_dir)
    if not audio_files:
        print(f"No audio files found in {args.data_dir}. Please run 'src/data/download.py' first.")
        return

    male_list, female_list = categorize_files_by_gender(audio_files)
    balanced_files = create_balanced_dataset(male_list, female_list, seed=RANDOM_SEED)
    
    # 2. Extract Features
    print("Extracting features (this may take a while)...")
    data = []
    labels = []
    
    # Check if features are already saved
    features_path = os.path.join(PROCESSED_DATA_DIR, "gender_features.csv")
    if os.path.exists(features_path):
        print(f"Loading features from {features_path}...")
        df = pd.read_csv(features_path)
        X = df.drop("label", axis=1)
        y = df["label"]
    else:
        for file in balanced_files:
            try:
                # Preprocess and Extract
                y_audio, sr = preprocess_audio_file(file)
                features = extract_features(y_audio, sr)
                
                # Add label
                if "female" in file.lower():
                    label = "female"
                else:
                    label = "male"
                
                data.append(features)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file}: {e}")

        df = pd.DataFrame(data)
        df["label"] = labels
        
        # Save features
        df.to_csv(features_path, index=False)
        X = df.drop("label", axis=1)
        y = df["label"]

    # 3. Prepare Data
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_SEED, stratify=y_encoded)
    
    # Impute and Scale
    imputer = SimpleImputer(strategy='mean')
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)
    
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # 4. Train Model
    print(f"Training {args.model} model...")
    if args.model == "knn":
        model = get_knn_model()
    elif args.model == "svm":
        model = get_svm_model()
    elif args.model == "xgboost":
        model = get_xgboost_model()
    elif args.model == "adaboost":
        model = get_adaboost_model()
        
    model.fit(X_train, y_train)
    
    # 5. Evaluate
    print("Evaluating model...")
    y_pred = model.predict(X_test)
    y_pred_prob = model.predict_proba(X_test)[:, 1]
    
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # Plots
    # plot_confusion_matrix(y_test, y_pred, le.classes_, title=f"Confusion Matrix - {args.model.upper()}")
    # plot_roc_curve(y_test, y_pred_prob, title=f"ROC Curve - {args.model.upper()}")
    print("Done!")

if __name__ == "__main__":
    main()
