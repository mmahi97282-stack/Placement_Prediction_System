import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder
from sklearn.impute import SimpleImputer

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Seamlessly parses both regular CSV files and pipe-delimited boxed grid ASCII tables.
# 2. Ensures data types and whitespace are cleaned before applying ordinal transformations.
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
# MAIN SCRIPT EXECUTION: MODULE 2 (CLEANING & ORDINAL ENCODING)
# WHY WRITTEN:
# Ordinal Encoding is specifically engineered for variables where order, rank, or hierarchy holds significance:
#   1. Dimensionality Control: Unlike One-Hot Encoding (which creates k new binary columns),
#      Ordinal Encoding retains a single column per feature, preventing the 'Curse of Dimensionality'.
#   2. Mathematical Monotonicity: Models with tree architectures (e.g. Decision Trees, Gradient Boosting)
#      can find optimal thresholds with single split points along ordinal integer axes.
#   3. Multi-Feature Processing: Applies scikit-learn's OrdinalEncoder across all categorical columns simultaneously,
#      prefixing output columns with 'Ordinal_' for unmistakable feature provenance.
# ====================================================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print("     MODULE 2: CLEANING & ORDINAL ENCODING PIPELINE")
    print("=" * 70)

    try:
        data = load_dataset(DATASET_PATH)

        print(f"[INFO] Raw Input Dataset Loaded. Dimensions: {data.shape[0]} Rows x {data.shape[1]} Columns")
        print_formatted_table(data, title="Raw Input Dataset Sample (First 5 Rows)")

        # Step 1: Normalize string whitespace
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype(str).str.strip()

        # Step 2: Remove duplicate rows
        before_rows = data.shape[0]
        data = data.drop_duplicates()
        after_rows = data.shape[0]
        duplicates_removed = before_rows - after_rows
        print(f"\n[INFO] Duplicate Records Removed: {duplicates_removed}")

        # Step 3: Partition numerical vs categorical features
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

        # Step 4: Missing Value Imputation
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy="mean")
            data[num_cols] = num_imputer.fit_transform(data[num_cols])

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

        # Step 5: Ordinal Encoding
        # WHY WRITTEN: Encodes categories into [0, 1, 2, ...] while preserving single-column feature width.
        if len(cat_cols) > 0:
            ordinal_encoder = OrdinalEncoder()
            encoded_values = ordinal_encoder.fit_transform(data[cat_cols])
            encoded_df = pd.DataFrame(
                encoded_values,
                columns=["Ordinal_" + col for col in cat_cols]
            )

            final_output = pd.concat(
                [data[num_cols].reset_index(drop=True), encoded_df.reset_index(drop=True)],
                axis=1
            )
        else:
            final_output = data.copy()

        total_missing_after = final_output.isnull().sum().sum()

        # Step 6: Dual Storage Output Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_ordinal_encode_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_ordinal_encode_M2.csv")

        save_table_csv(final_output, output_file_outputs)
        save_table_csv(final_output, output_file_dataset)

        print_formatted_table(final_output, title="Cleaned & Ordinal Encoded Dataset (First 5 Rows)")

        print("\n" + "-" * 70)
        print("                  SUMMARY OF ORDINAL ENCODED DATASET")
        print("-" * 70)
        print(f"  • Final Output Shape       : {final_output.shape[0]} Rows x {final_output.shape[1]} Columns")
        print(f"  • Duplicates Cleaned       : {duplicates_removed}")
        print(f"  • Encoded Categorical Cols  : {', '.join(cat_cols)}")
        print(f"  • Primary Output Saved To  : {output_file_outputs}")
        print(f"  • Dataset Folder Sync To   : {output_file_dataset}")
        print("=" * 70)
        print("[SUCCESS] Module 2: Ordinal Encoding Pipeline Completed Successfully.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

