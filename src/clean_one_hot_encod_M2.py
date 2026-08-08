import os
import io
import pandas as pd
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Allows loading datasets whether stored as standard CSVs or as human-readable pipe-delimited grid tables.
# 2. Removes ASCII border rows and casts data cleanly so preprocessing imputation works without string parsing issues.
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
# 1. Standard pandas to_csv creates unaligned text where columns are jagged across lines.
# 2. This custom writer dynamically calculates maximum cell widths across all columns and pads entries
#    with spaces, creating an aesthetically pleasing, table-aligned CSV file that is effortless to read in text editors.
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
# 1. Renders a formatted ASCII border grid in the console for instant visual verification of output data.
# 2. Displays headers, boundaries, and dynamic widths so developers can inspect encoding results in real-time.
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
# MAIN SCRIPT EXECUTION: MODULE 2 (CLEANING & ONE-HOT ENCODING)
# WHY WRITTEN:
# Machine Learning models (Logistic Regression, Random Forest, Neural Networks) operate strictly on numerical matrices.
# Categorical string features like 'Department' (CSE, ECE, Civil) cannot be processed mathematically without encoding:
#   1. Data Cleaning: Trims whitespace from all text cells to prevent duplicates like 'CSE ' vs 'CSE'.
#   2. Missing Value Imputation:
#      - SimpleImputer(strategy='mean'): Fills missing numerical values with the feature mean, preserving central tendency.
#      - SimpleImputer(strategy='most_frequent'): Fills missing categorical values with the mode category.
#   3. One-Hot Encoding:
#      - Converts N distinct nominal categories into N binary (0 or 1) dummy indicator columns.
#      - WHY NOT LABEL ENCODING HERE? Label encoding imposes an artificial mathematical hierarchy
#        (e.g., Civil=1, CSE=2, ECE=3 implies ECE > CSE > Civil), which distorts distance-based models.
#        One-Hot Encoding ensures completely neutral, independent feature representation.
#   4. Dual Persistence: Saves the processed dataset to both outputs/cleaned_data/ and dataset/ directories.
# ====================================================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print("   MODULE 2: DATA CLEANING & ONE-HOT ENCODING PIPELINE")
    print("=" * 70)

    try:
        data = load_dataset(DATASET_PATH)

        print(f"[INFO] Raw Input Dataset Loaded. Shape: {data.shape[0]} Rows x {data.shape[1]} Columns")
        print_formatted_table(data, title="Raw Input Dataset Sample (First 5 Rows)")

        # Step 1: Whitespace Normalization
        # WHY WRITTEN: Extra trailing or leading spaces create false duplicate categories in categorical encoders.
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype(str).str.strip()

        # Step 2: Duplicate Record Removal
        # WHY WRITTEN: Duplicate student records bias gradient descent and distort validation metrics.
        before_rows = data.shape[0]
        data = data.drop_duplicates()
        after_rows = data.shape[0]
        duplicates_removed = before_rows - after_rows
        print(f"\n[INFO] Duplicate Records Removed: {duplicates_removed}")

        # Step 3: Numerical and Categorical Column Separation
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

        # Step 4: Missing Value Imputation
        # WHY WRITTEN: Scikit-learn estimators raise ValueError when encountering NaN values.
        if len(num_cols) > 0:
            num_imputer = SimpleImputer(strategy="mean")
            data[num_cols] = num_imputer.fit_transform(data[num_cols])

        if len(cat_cols) > 0:
            cat_imputer = SimpleImputer(strategy="most_frequent")
            data[cat_cols] = cat_imputer.fit_transform(data[cat_cols])

        # Step 5: One-Hot Encoding Transformation
        # WHY WRITTEN: Creates orthogonal binary indicator columns for each category without false ordinal rank.
        if len(cat_cols) > 0:
            encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
            encoded_values = encoder.fit_transform(data[cat_cols])
            encoded_df = pd.DataFrame(encoded_values, columns=encoder.get_feature_names_out(cat_cols))

            encoded_df.reset_index(drop=True, inplace=True)
            numeric_df = data[num_cols].reset_index(drop=True)
            # Combine numeric features with one-hot encoded dummy features
            final_output = pd.concat([numeric_df, encoded_df], axis=1)
        else:
            final_output = data.copy()

        # Step 6: Dual Storage Output Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_one_hot_encoding_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_one_hot_encoding_M2.csv")

        save_table_csv(final_output, output_file_outputs)
        save_table_csv(final_output, output_file_dataset)

        print_formatted_table(final_output.iloc[:, :10], title="Cleaned & One-Hot Encoded Dataset (First 10 Columns)")

        print("\n" + "-" * 70)
        print("                  SUMMARY OF ONE-HOT ENCODED DATASET")
        print("-" * 70)
        print(f"  • Final Output Dataset Shape : {final_output.shape[0]} Rows x {final_output.shape[1]} Columns")
        print(f"  • Duplicate Rows Cleaned    : {duplicates_removed}")
        print(f"  • Encoded Categorical Cols  : {', '.join(cat_cols)}")
        print(f"  • Output Location (Outputs) : {output_file_outputs}")
        print(f"  • Output Location (Dataset) : {output_file_dataset}")
        print("=" * 70)
        print("[SUCCESS] Module 2: One-Hot Encoding Pipeline Completed Successfully.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

