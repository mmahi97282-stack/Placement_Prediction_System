import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib
# Headless Matplotlib Backend: Prevents GUI window errors during automated background execution
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# Safely loads standard CSVs as well as boxed grid ASCII/pipe tables with whitespace trimming and type casting.
# ====================================================================================================
def load_dataset(filepath):
    """
    Safely loads dataset supporting both standard comma-separated CSVs
    and human-readable boxed grid ASCII/pipe tables.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Dataset file not found at: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    if "|" in content:
        lines = [l for l in content.splitlines() if not l.strip().startswith("+") and l.strip()]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", skipinitialspace=True).dropna(how="all", axis=1)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        return df
    else:
        df = pd.read_csv(filepath, skipinitialspace=True)
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        return df


# ====================================================================================================
# MAIN SCRIPT EXECUTION: MODULE 2 (FEATURE SCALING, STANDARDIZATION & L1 NORMALIZATION)
# WHY WRITTEN:
# Feature scaling adjusts numerical attributes to common scales so that features with large numerical
# values (e.g. TenthPercentage ~80) do not overpower smaller magnitude features (e.g. Backlogs ~1-3):
#   1. StandardScaler: Centers data around zero with unit standard deviation ($z = (x-\mu)/\sigma$).
#   2. MinMaxScaler: Transforms features linearly into $[0, 1]$ interval ($x' = (x-min)/(max-min)$).
#   3. L1 Normalizer: Divides each sample by its Manhattan norm ($\|x\|_1 = \sum |x_i| = 1$),
#      useful for probability mass distributions and sparse linear models (Lasso / L1 regularization).
# ====================================================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
    EDA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "EDA_Analysis_outputs")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print(" MODULE 2: FEATURE SCALING, STANDARDIZATION & L1 NORMALIZATION")
    print("=" * 70)

    try:
        df = load_dataset(DATASET_PATH)
        print(f"[INFO] Initial Dataset Loaded. Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns")

        # Step 1: Deduplication
        duplicates_removed = df.duplicated().sum()
        df = df.drop_duplicates()

        # Step 2: Categorical Imputation
        categorical_columns = df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].fillna(df[col].mode()[0])

        # Step 3: Numerical Imputation
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].mean())

        # Step 4: Standardization (Z-score Scaling)
        # WHY WRITTEN: Centers data with mean=0 and unit variance for gradient descent optimization.
        standard_scaler = StandardScaler()
        standardized = standard_scaler.fit_transform(df[numerical_columns])
        for i, col in enumerate(numerical_columns):
            df[col + "_Standardized"] = standardized[:, i]

        # Step 5: Min-Max Scaling (0.0 to 1.0)
        # WHY WRITTEN: Constrains all numerical variables to a bounded [0, 1] range.
        minmax_scaler = MinMaxScaler()
        scaled = minmax_scaler.fit_transform(df[numerical_columns])
        for i, col in enumerate(numerical_columns):
            df[col + "_Scaled"] = scaled[:, i]

        # Step 6: L1 Manhattan Normalization
        # WHY WRITTEN: Forces the sum of absolute values for each student record to equal 1.0.
        normalizer = Normalizer(norm='l1')
        normalized = normalizer.fit_transform(df[numerical_columns])
        for i, col in enumerate(numerical_columns):
            df[col + "_Normalized"] = normalized[:, i]

        # Step 7: Dual Output Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_minmax_stand_norma_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_minmax_stand_norma_M2.csv")

        df.to_csv(output_file_outputs, index=False)
        df.to_csv(output_file_dataset, index=False)

        total_missing_after = df.isnull().sum().sum()

        print("\n" + "-" * 70)
        print("                  SUMMARY OF CLEANED DATASET")
        print("-" * 70)
        print(f"  • Final Dataset Shape      : {df.shape[0]} Rows x {df.shape[1]} Columns")
        print(f"  • Duplicates Removed       : {duplicates_removed}")
        print(f"  • Missing Values Remaining : {total_missing_after}")
        print(f"  • Transformations Applied  : Standardized, MinMax Scaled, L1 Normalized")
        print(f"  • Primary Output Saved To  : {output_file_outputs}")
        print("=" * 70)
        print("[SUCCESS] Feature Scaling Pipeline Execution Completed.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()