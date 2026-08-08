import os
import io
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# ====================================================================================================
# CONFIGURATION & DIRECTORY MANAGEMENT
# WHY WRITTEN:
# 1. Models and transformers (scalers, encoders) must be serialized to a dedicated models/ directory
#    so the Flask web application can load them instantly during real-time inference without retraining.
# 2. Target candidate detection guarantees resilience against minor column naming variations in user CSVs.
# ====================================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

TARGET_CANDIDATES = ["PlacementStatus", "Placement", "Status", "Placed"]

# ====================================================================================================
# FUNCTION: load_data
# WHY WRITTEN:
# 1. Accommodates standard comma-separated CSVs and human-readable pipe-delimited boxed grid ASCII tables.
# 2. Identifies the ground-truth target column from known candidates and validates dataset non-emptiness.
# ====================================================================================================
def load_data():
    """
    Loads placement training data and identifies the target classification label.
    """
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"[ERROR] Training dataset not found at path: {DATASET_PATH}")
    
    with open(DATASET_PATH, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "|" in content:
        lines = [l for l in content.splitlines() if not l.strip().startswith("+") and l.strip()]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", skipinitialspace=True).dropna(how="all", axis=1)
        df.columns = [c.strip() for c in df.columns]
    else:
        df = pd.read_csv(DATASET_PATH, skipinitialspace=True)
    
    for col in df.columns:
        if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
            df[col] = df[col].astype(str).str.strip()
            try:
                df[col] = pd.to_numeric(df[col])
            except (ValueError, TypeError):
                pass
                
    target = next((c for c in TARGET_CANDIDATES if c in df.columns), None)
    if target is None:
        raise ValueError(
            f"[ERROR] Could not find target column. Expected one of {TARGET_CANDIDATES}. "
            f"Found: {df.columns.tolist()}"
        )
    return df, target


# ====================================================================================================
# FUNCTION: preprocess
# WHY WRITTEN:
# Preprocessing prepares raw student records for machine learning estimation:
#   1. Deduplication: Removes identical records to avoid artificial over-fitting on duplicate samples.
#   2. Missing Value Imputation:
#      - Median Imputation for numeric attributes: Median is robust to outliers and skewed distributions.
#      - Mode Imputation for categorical attributes: Uses the most frequent class for missing categories.
#   3. Label Encoding: Maps string categories to integer indices and stores the fitted encoders in a dictionary
#      for runtime decoding.
#   4. StandardScaler: Fits a standard normal z-score transformer ($Z = (X-\mu)/\sigma$) across all predictor columns.
# ====================================================================================================
def preprocess(df, target):
    """
    Transforms raw dataframe into standardized feature matrix X and target array y.
    """
    df = df.drop_duplicates().copy()

    # Step 1: Impute missing numerical & categorical values
    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0])

    # Step 2: Categorical Feature Encoding
    encoders = {}
    cat_cols = [c for c in df.columns if not pd.api.types.is_numeric_dtype(df[c])]
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))
        encoders[col] = le

    # Step 3: Feature & Target Segregation
    X = df.drop(columns=[target])
    y = df[target]

    # Step 4: Feature Standardization
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, y, scaler, encoders, X.columns.tolist()


# ====================================================================================================
# FUNCTION: train_and_evaluate
# WHY WRITTEN:
# Evaluates a diverse multi-paradigm ensemble of 4 distinct machine learning algorithms:
#   1. Logistic Regression: Linear baseline offering fast probabilistic odds interpretation.
#   2. Decision Tree: Non-linear rule-based partitioning with human-interpretable branch decisions.
#   3. Random Forest (Ensemble Bagging): Combines 100 de-correlated decision trees with bootstrap aggregation
#      to reduce variance and eliminate single-tree over-fitting.
#   4. K-Nearest Neighbors (Instance-Based): Classifies students based on Euclidean proximity to similar peers.
# Metrics: Computes Accuracy, Precision, Recall, and F1-Score (harmonic mean of precision & recall).
# Best Model Selection: Automatically tracks and returns the highest-accuracy classifier.
# ====================================================================================================
def train_and_evaluate(X_train, X_test, y_train, y_test):
    """
    Trains and compares multiple classification models across 4 performance metrics.
    """
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Random Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=max(1, min(3, len(X_train)))),
    }

    results = {}
    best_name, best_model, best_acc = None, None, -1.0

    print("\n" + "-" * 75)
    print(f"{'Model Architecture':<24} | {'Accuracy':<10} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
    print("-" * 75)

    for name, model in models.items():
        # Fit model on training split
        model.fit(X_train, y_train)
        preds = model.predict(X_test)

        # Multi-metric classification evaluation
        acc = accuracy_score(y_test, preds)
        prec = precision_score(y_test, preds, average="weighted", zero_division=0)
        rec = recall_score(y_test, preds, average="weighted", zero_division=0)
        f1 = f1_score(y_test, preds, average="weighted", zero_division=0)

        results[name] = {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}
        print(f"{name:<24} | {acc:<10.4f} | {prec:<10.4f} | {rec:<10.4f} | {f1:<10.4f}")

        # Track best performing model
        if acc > best_acc or best_model is None:
            best_name, best_model, best_acc = name, model, acc

    print("-" * 75)
    return results, best_name, best_model


# ====================================================================================================
# MAIN SCRIPT EXECUTION
# WHY WRITTEN:
# Coordinates the entire machine learning lifecycle:
#   1. Loads dataset and validates target column presence.
#   2. Preprocesses and scales features to standard normal distributions.
#   3. Performs an 80/20 train/test split with stratification on target labels.
#   4. Benchmarks 4 model architectures and selects the champion model.
#   5. Serializes model weights, scalers, encoders, and feature column schemas to disk using Joblib.
# ====================================================================================================
def main():
    print("=" * 75)
    print("             MACHINE LEARNING MODEL TRAINING PIPELINE")
    print("=" * 75)

    try:
        # Step 1: Load data
        df, target = load_data()
        print(f"[INFO] Dataset Loaded ({df.shape[0]} rows x {df.shape[1]} cols). Target: '{target}'")

        # Step 2: Preprocess and extract feature arrays
        X, y, scaler, encoders, feature_names = preprocess(df, target)

        # Step 3: Train / Test Split
        # WHY WRITTEN: Stratification ensures both train and test partitions maintain identical class balance ratios.
        stratify_arg = y if len(np.unique(y)) > 1 and min(pd.Series(y).value_counts()) > 1 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=stratify_arg
        )

        # Step 4: Model Training & Benchmarking
        results, best_name, best_model = train_and_evaluate(X_train, X_test, y_train, y_test)

        # Step 5: Save Artifacts for Flask Inference
        # WHY WRITTEN: The Flask web app requires exact pipeline artifacts to perform live student predictions.
        joblib.dump(best_model, os.path.join(MODEL_DIR, "placement_model.pkl"))
        joblib.dump(scaler, os.path.join(MODEL_DIR, "scaler.pkl"))
        joblib.dump(encoders, os.path.join(MODEL_DIR, "encoders.pkl"))
        joblib.dump(feature_names, os.path.join(MODEL_DIR, "feature_names.pkl"))

        print("\n" + "=" * 75)
        print("                    TRAINING SUMMARY & ARTIFACTS")
        print("=" * 75)
        print(f"  • Selected Best Model       : {best_name}")
        print(f"  • Validation Accuracy       : {results[best_name]['accuracy'] * 100:.2f}%")
        print(f"  • Weighted F1-Score         : {results[best_name]['f1']:.4f}")
        print(f"  • Serialized Artifacts Path : {MODEL_DIR}")
        print(f"    - placement_model.pkl (Trained Classifier)")
        print(f"    - scaler.pkl (StandardScaler)")
        print(f"    - encoders.pkl (LabelEncoders)")
        print(f"    - feature_names.pkl (Input Schema)")
        print("=" * 75)
        print("[SUCCESS] Machine Learning Model Training Completed Successfully.")
        print("=" * 75)

    except Exception as ex:
        print(f"[ERROR] Model training pipeline encountered an error: {ex}")


if __name__ == "__main__":
    main()

