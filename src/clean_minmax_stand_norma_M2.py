import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler, Normalizer
import matplotlib
# Headless Matplotlib Backend
# WHY WRITTEN: Prevents GUI window errors in background execution and web server production tasks.
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
# FUNCTION: save_table_csv
# WHY WRITTEN:
# Formats pandas DataFrames into clean, column-aligned CSV files with uniform spacing.
# ====================================================================================================
def save_table_csv(df_to_save, filepath):
    """
    Writes a pandas DataFrame to disk formatted as a clean, column-aligned CSV.
    """
    cols = [str(c) for c in df_to_save.columns]
    widths = {c: max([len(str(val)) for val in df_to_save[c]] + [len(c)]) for c in cols}
    header_line = " , ".join([f"{c:<{widths[c]}}" for c in cols])
    data_rows = [header_line]
    for _, row in df_to_save.iterrows():
        row_line = " , ".join([f"{str(row[c]):<{widths[c]}}" for c in cols])
        data_rows.append(row_line)
    formatted_content = "\n".join(data_rows) + "\n"
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_content)


# ====================================================================================================
# FUNCTION: print_formatted_table
# WHY WRITTEN:
# Prints a structured ASCII bordered table to standard output for real-time inspection.
# ====================================================================================================
def print_formatted_table(df_sample, title="Dataset Sample", max_rows=5, padding=4):
    """
    Renders an ASCII grid table for console viewing.
    """
    if df_sample.empty:
        print(f"\n--- {title} --- [Empty DataFrame]")
        return
    sample = df_sample.head(max_rows).copy()
    cols = [str(c) for c in sample.columns]
    widths = {c: max([len(str(val)) for val in sample[c]] + [len(c)]) + padding for c in sample.columns}
    
    border = "+" + "+".join(["-" * widths[c] for c in cols]) + "+"
    header = "|" + "|".join([f" {c:<{widths[c]-1}}" for c in cols]) + "|"
    
    print(f"\n--- {title} ---")
    print(border)
    print(header)
    print(border)
    for _, row in sample.iterrows():
        row_str = "|" + "|".join([f" {str(row[c]):<{widths[c]-1}}" for c in sample.columns]) + "|"
        print(row_str)
    print(border)


# ====================================================================================================
# MAIN SCRIPT EXECUTION: MODULE 2 (FEATURE SCALING, STANDARDIZATION & L2 NORMALIZATION)
# WHY WRITTEN:
# Machine Learning models (specifically Gradient Descent optimizers, KNN, and SVMs) are sensitive to feature scales:
#   - Problem: CGPA is scaled [0-10], AptitudeScore is scaled [0-100], and Backlogs is [0-10].
#     Without scaling, AptitudeScore's raw magnitude dominates distance metrics ($d = \sqrt{\sum (x_i - y_i)^2}$).
#   1. Z-score Standardization (StandardScaler):
#      - Formula: z = (x - \mu) / \sigma
#      - Centers features to mean=0 and variance=1. Ideal for algorithms assuming Gaussian distributions.
#   2. Min-Max Scaling (MinMaxScaler):
#      - Formula: x_scaled = (x - x_min) / (x_max - x_min)
#      - Bounded strictly within [0.0, 1.0]. Essential for neural network activation functions (Sigmoid/ReLU).
#   3. L2 Vector Normalization (Normalizer):
#      - Formula: x_norm = x / \sqrt{\sum x_i^2}
#      - Rescales each student row vector to unit Euclidean norm (magnitude = 1.0), ideal for Cosine similarity.
#   4. Preprocessed Histogram Visualization:
#      - Generates multi-panel histogram grid saved to outputs/EDA_Analysis_outputs/ to confirm post-scaling distributions.
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
    print(" MODULE 2: FEATURE SCALING, STANDARDIZATION & L2 NORMALIZATION")
    print("=" * 70)

    try:
        df = load_dataset(DATASET_PATH)
        print(f"[INFO] Raw Input Dataset Loaded. Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns")
        print_formatted_table(df, title="Raw Input Dataset Sample (First 5 Rows)")

        # Step 1: Deduplication
        duplicates_removed = df.duplicated().sum()
        df = df.drop_duplicates()
        print(f"\n[INFO] Duplicate Records Removed: {duplicates_removed}")

        # Step 2: Categorical Whitespace Stripping & Imputation
        categorical_columns = df.select_dtypes(include=['object', 'string']).columns
        for col in categorical_columns:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].fillna(df[col].mode()[0])

        # Step 3: Numerical Imputation
        numerical_columns = df.select_dtypes(include=['int64', 'float64']).columns
        for col in numerical_columns:
            df[col] = df[col].fillna(df[col].mean())

        # Step 4: Standardization (Z-score: Mean=0, Std=1)
        # WHY WRITTEN: Eliminates scale disparities across differently unitized academic indicators.
        standard_scaler = StandardScaler()
        standardized = standard_scaler.fit_transform(df[numerical_columns])
        for i, col in enumerate(numerical_columns):
            df[col + "_Standardized"] = standardized[:, i]

        # Step 5: Min-Max Normalization (Range: [0, 1])
        # WHY WRITTEN: Compresses all features into uniform bounded interval [0, 1].
        minmax_scaler = MinMaxScaler()
        scaled = minmax_scaler.fit_transform(df[numerical_columns])
        for i, col in enumerate(numerical_columns):
            df[col + "_Scaled"] = scaled[:, i]

        # Step 6: L2 Unit Vector Normalization
        # WHY WRITTEN: Projects individual student record vectors onto a unit hypersphere for direction-based distance analysis.
        normalizer = Normalizer(norm='l2')
        normalized = normalizer.fit_transform(df[numerical_columns])
        for i, col in enumerate(numerical_columns):
            df[col + "_Normalized"] = normalized[:, i]

        # Step 7: Dual Storage Output Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_minmax_stand_norma_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_minmax_stand_norma_M2.csv")

        save_table_csv(df, output_file_outputs)
        save_table_csv(df, output_file_dataset)

        print_formatted_table(df.iloc[:, :8], title="Preprocessed & Scaled Dataset Preview (First 8 Features)")

        # Step 8: Multi-Feature Histogram Chart
        try:
            df[numerical_columns].hist(figsize=(12, 10), bins=15, color='#3498db', edgecolor='black', alpha=0.8)
            plt.suptitle("Histograms of Continuous Numerical Features (Post-Cleaning)", fontsize=14, fontweight='bold')
            plt.tight_layout()
            chart_path = os.path.join(EDA_OUTPUT_DIR, "Histogram_Preprocessed.png")
            plt.savefig(chart_path, dpi=300)
            plt.close()
            print(f"\n[INFO] Preprocessed Feature Histogram saved to: {chart_path}")
        except Exception as e:
            print(f"\n[WARNING] Could not save histogram plot: {e}")

        total_missing_after = df.isnull().sum().sum()

        print("\n" + "-" * 70)
        print("                  SUMMARY OF SCALED & NORMALIZED DATASET")
        print("-" * 70)
        print(f"  • Final Output Shape        : {df.shape[0]} Rows x {df.shape[1]} Columns")
        print(f"  • Transformations Engineered: Standardized (_Standardized), MinMax Scaled (_Scaled), L2 Normalized (_Normalized)")
        print(f"  • Primary Output Saved To   : {output_file_outputs}")
        print(f"  • Dataset Folder Sync To    : {output_file_dataset}")
        print("=" * 70)
        print("[SUCCESS] Module 2: Feature Scaling & Normalization Pipeline Completed.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

