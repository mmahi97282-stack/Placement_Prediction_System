import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.impute import SimpleImputer

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Loads datasets supporting both standard CSVs and human-readable boxed grid ASCII/pipe tables.
# 2. Strips whitespace, removes decorative row boundaries, and cleanly parses numeric and categorical columns.
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
# Formats pandas DataFrames into column-aligned CSV files with uniform spacing for terminal & editor inspection.
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
# Renders a formatted ASCII bordered table to standard output for immediate visual validation.
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
# MAIN SCRIPT EXECUTION: MODULE 2 (CLEANING & TARGET MEAN ENCODING)
# WHY WRITTEN:
# Target Mean Encoding (Mean Response Encoding) replaces each categorical level with the expected conditional
# probability or mean of the target variable for that level: E[PlacementStatus | Department = 'CSE']:
#   1. High-Cardinality Compression: Compresses complex categorical variables with hundreds of levels into a single continuous float.
#   2. Strong Predictive Signal: Embeds the empirical posterior relationship between features and the target directly.
#   3. Numerical Representation: Linear and tree models can directly exploit the monotonic placement likelihood ranking.
#   4. Target Independence: Automatically detects and encodes the primary target candidate ('PlacementStatus', 'Placement', 'Placed').
# ====================================================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print("     MODULE 2: CLEANING & TARGET ENCODING PIPELINE")
    print("=" * 70)

    try:
        data = load_dataset(DATASET_PATH)

        print(f"[INFO] Raw Input Dataset Loaded. Dimensions: {data.shape[0]} Rows x {data.shape[1]} Columns")
        print_formatted_table(data, title="Raw Input Dataset Sample (First 5 Rows)")

        # Step 1: Whitespace Normalization
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype(str).str.strip()

        # Step 2: Duplicate Record Removal
        before_rows = data.shape[0]
        data = data.drop_duplicates()
        after_rows = data.shape[0]
        duplicates_removed = before_rows - after_rows
        print(f"\n[INFO] Duplicate Records Removed: {duplicates_removed}")

        # Step 3: Numerical and Categorical Partitioning
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

        # Step 4: Missing Value Imputation
        for col in num_cols:
            data[col] = data[col].fillna(data[col].mean())

        for col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        # Step 5: Identify Target Variable
        TARGET_CANDIDATES = ["PlacementStatus", "Placement", "Status", "Placed"]
        target_column = next((c for c in TARGET_CANDIDATES if c in data.columns), None)
        if target_column is None:
            target_column = data.columns[-1]

        print(f"\n[INFO] Target Dependent Variable Identified: '{target_column}'")

        # Convert target to numeric 0/1 indicator if stored as string
        if not pd.api.types.is_numeric_dtype(data[target_column]):
            data["target_numeric"] = LabelEncoder().fit_transform(data[target_column].astype(str))
            calc_target = "target_numeric"
        else:
            calc_target = target_column

        # Step 6: Target Mean Encoding Calculation
        # WHY WRITTEN: Groups each category by target mean (e.g. CSE average placement rate = 0.85).
        target_encoded_df = pd.DataFrame()
        for col in cat_cols:
            if col != target_column:
                mean_encoding = data.groupby(col)[calc_target].mean()
                target_encoded_df["Target_" + col] = data[col].map(mean_encoding)
                print(f"  • Feature '{col}' mapped to target conditional probabilities:")
                for cat_name, rate in mean_encoding.items():
                    print(f"    - {cat_name:<15} : {rate:.3f} placement mean")

        if "target_numeric" in data.columns:
            data.drop(columns=["target_numeric"], inplace=True)

        final_output = pd.concat(
            [data[num_cols].reset_index(drop=True), target_encoded_df.reset_index(drop=True)],
            axis=1
        )

        total_missing_after = final_output.isnull().sum().sum()

        # Step 7: Dual Storage Output Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_target_encode_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_target_encode_M2.csv")

        save_table_csv(final_output, output_file_outputs)
        save_table_csv(final_output, output_file_dataset)

        print_formatted_table(final_output, title="Cleaned & Target Encoded Dataset (First 5 Rows)")

        print("\n" + "-" * 70)
        print("                  SUMMARY OF TARGET ENCODED DATASET")
        print("-" * 70)
        print(f"  • Final Output Dataset Shape : {final_output.shape[0]} Rows x {final_output.shape[1]} Columns")
        print(f"  • Duplicates Cleaned        : {duplicates_removed}")
        print(f"  • Encoded Categorical Cols  : {[c for c in cat_cols if c != target_column]}")
        print(f"  • Primary Output Saved To   : {output_file_outputs}")
        print(f"  • Dataset Folder Sync To    : {output_file_dataset}")
        print("=" * 70)
        print("[SUCCESS] Module 2: Target Encoding Pipeline Completed Successfully.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

