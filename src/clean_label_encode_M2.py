import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Accommodates standard comma-delimited CSVs and pipe-delimited ASCII grid datasets.
# 2. Ensures data is cleanly parsed into numeric and string columns before label encoding.
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
# Formats pandas DataFrames into aesthetically aligned CSV files with uniform column widths.
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
# Prints a structured ASCII bordered table to standard output for visual inspection.
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
# MAIN SCRIPT EXECUTION: MODULE 2 (CLEANING & LABEL ENCODING)
# WHY WRITTEN:
# Label Encoding maps every distinct categorical value within a column to an integer from 0 to k-1:
#   1. Memory Efficiency: Replaces multi-byte text strings (e.g. 'Mechanical', 'Not Placed') with compact integers.
#   2. Tree-Based Compatibility: Tree-based ensemble classifiers (Random Forest, Decision Tree, XGBoost)
#      can split numerical label encodings effectively without requiring memory-heavy dimensionality expansion.
#   3. Target Label Transformation: LabelEncoder is the industry standard for converting target class labels
#      ('Placed' -> 1, 'Not Placed' -> 0) into binary targets for loss functions (e.g. binary cross-entropy).
#   4. Reversibility: Fitted LabelEncoder objects preserve mapping dictionaries, allowing inverse transformation
#      during production inference to deliver human-readable predictions to end users.
# ====================================================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print("     MODULE 2: CLEANING & LABEL ENCODING PIPELINE")
    print("=" * 70)

    try:
        data = load_dataset(DATASET_PATH)

        print(f"[INFO] Raw Input Dataset Loaded. Dimensions: {data.shape[0]} Rows x {data.shape[1]} Columns")
        print_formatted_table(data, title="Raw Input Dataset Sample (First 5 Rows)")

        # Step 1: Whitespace Normalization
        # WHY WRITTEN: Ensures 'Yes ' and 'Yes' are treated as the same class rather than distinct classes.
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype(str).str.strip()

        # Step 2: Missingness Diagnostics
        missing_before = data.isnull().sum().sum()
        print(f"\n[INFO] Total Missing Values Detected Before Imputation: {missing_before}")

        # Step 3: Duplicate Record Elimination
        # WHY WRITTEN: Duplicate samples skew training distributions and inflate model evaluation metrics.
        before_rows = data.shape[0]
        data = data.drop_duplicates()
        after_rows = data.shape[0]
        duplicates_removed = before_rows - after_rows
        print(f"[INFO] Duplicate Records Removed: {duplicates_removed}")

        # Step 4: Numerical and Categorical Partitioning
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

        # Step 5: Imputation Pipeline
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy="mean")
            data[num_cols] = num_imputer.fit_transform(data[num_cols])

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

        # Step 6: Label Encoding Pipeline
        # WHY WRITTEN: Encodes each unique string to a distinct integer index [0, 1, 2, ...].
        label_encoders = {}
        for col in cat_cols:
            encoder = LabelEncoder()
            data[col] = encoder.fit_transform(data[col].astype(str))
            label_encoders[col] = encoder
            # Print mapping for developer transparency
            mapping = {val: idx for idx, val in enumerate(encoder.classes_)}
            print(f"  • Feature '{col}' encoded with {len(encoder.classes_)} classes: {mapping}")

        total_missing_after = data.isnull().sum().sum()

        # Step 7: Dual-Location Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_label_encode_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_label_encode_M2.csv")

        save_table_csv(data, output_file_outputs)
        save_table_csv(data, output_file_dataset)

        print_formatted_table(data, title="Cleaned & Label Encoded Dataset Preview (First 5 Rows)")

        print("\n" + "-" * 70)
        print("                  SUMMARY OF LABEL ENCODED DATASET")
        print("-" * 70)
        print(f"  • Final Output Shape       : {data.shape[0]} Rows x {data.shape[1]} Columns")
        print(f"  • Duplicates Cleaned       : {duplicates_removed}")
        print(f"  • Missing Values Remaining : {total_missing_after}")
        print(f"  • Primary Output Saved To  : {output_file_outputs}")
        print(f"  • Dataset Folder Sync To   : {output_file_dataset}")
        print("=" * 70)
        print("[SUCCESS] Module 2: Label Encoding Pipeline Completed Successfully.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

