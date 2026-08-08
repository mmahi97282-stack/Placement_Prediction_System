import os
import io
import pandas as pd
import numpy as np
import matplotlib
# Set Matplotlib backend to 'Agg' to render charts headlessly without requiring a GUI window or display server.
# WHY WRITTEN: In web servers, terminal environments, and backend tasks, GUI backends (like Tkinter/Qt) crash with 'no display' errors.
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Datasets in this system can be stored either as standard comma-separated CSV files or as
#    human-readable boxed grid tables formatted with pipe delimiters ('|') and ASCII border lines ('+---').
# 2. This helper inspects the raw file text, detects pipe borders, removes decorative border rows,
#    and parses the data cleanly into a strongly-typed pandas DataFrame.
# 3. It strips extraneous leading and trailing whitespace from string columns and infers proper numeric types.
# ====================================================================================================
def load_dataset(filepath):
    """
    Safely load dataset supporting both standard comma-separated CSVs
    and human-readable boxed grid ASCII/pipe tables.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Dataset file not found at: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # WHY WRITTEN: If '|' is present in file content, the file is formatted as an ASCII boxed grid table.
    if "|" in content:
        # Filter out decorative horizontal border lines starting with '+'
        lines = [l for l in content.splitlines() if not l.strip().startswith("+") and l.strip()]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", skipinitialspace=True).dropna(how="all", axis=1)
        # Strip header column names of whitespace
        df.columns = [c.strip() for c in df.columns]
        # Clean values and attempt numeric type casting
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
                try:
                    df[col] = pd.to_numeric(df[col])
                except (ValueError, TypeError):
                    pass
        return df
    else:
        # Standard CSV fallback
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
# FUNCTION: print_formatted_table
# WHY WRITTEN:
# 1. Console outputs are often hard to read when column lengths vary.
# 2. This function dynamically calculates column widths based on maximum string lengths of both
#    the header and data cells, rendering a clean, perfectly aligned ASCII table.
# 3. Padding is applied to ensure visual separation between adjacent columns.
# ====================================================================================================
def print_formatted_table(df_sample, title="Dataset Sample", max_rows=5, padding=4):
    """
    Renders a clean ASCII grid table for console viewing with dynamically adjusted column widths.
    """
    if df_sample.empty:
        print(f"\n--- {title} --- [Empty DataFrame]")
        return
    
    # Take the top N rows for quick inspection
    sample = df_sample.head(max_rows).copy()
    cols = [str(c) for c in sample.columns]
    
    # Compute maximum width needed for each column (max of header name vs row values + padding)
    widths = {c: max([len(str(val)) for val in sample[c]] + [len(c)]) + padding for c in sample.columns}
    
    # Build top/bottom border and header line
    border = "+" + "+".join(["-" * widths[c] for c in cols]) + "+"
    header = "|" + "|".join([f" {c:<{widths[c]-1}}" for c in cols]) + "|"
    
    print(f"\n--- {title} ---")
    print(border)
    print(header)
    print(border)
    
    # Print individual formatted data rows
    for _, row in sample.iterrows():
        row_str = "|" + "|".join([f" {str(row[c]):<{widths[c]-1}}" for c in sample.columns]) + "|"
        print(row_str)
    print(border)


# ====================================================================================================
# MAIN SCRIPT EXECUTION
# WHY WRITTEN:
# Module 1 fulfills Step 1 (Load), Step 2 (Understand Structure), and Step 3 (Missingness & Stats)
# of the Placement Prediction Data Science Lifecycle:
#   - Step 1: Loads the placement dataset from disk.
#   - Step 2: Analyzes column metadata, shapes, datatypes, and displays sample rows.
#   - Step 3: Quantifies null values, checks duplicate rows, computes descriptive statistics,
#             and generates a high-resolution CGPA distribution histogram.
# ====================================================================================================
def main():
    # Setup directory paths dynamically relative to script location
    # WHY WRITTEN: Hardcoded absolute paths break across different computers or OS environments.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    EDA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "EDA_Analysis_outputs")
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("     MODULE 1: LOAD & UNDERSTAND PLACEMENT PREDICTION DATASET")
    print("=" * 70)

    try:
        # 1. Load the raw dataset
        df = load_dataset(DATASET_PATH)

        print(f"\n[INFO] Dataset File Path: {DATASET_PATH}")
        print(f"[INFO] Dataset Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns\n")

        # 2. Inspect Column Metadata and Data Types
        # WHY WRITTEN: Knowing whether a column is numeric (float/int) or categorical (object/string)
        # is necessary to determine which preprocessing (scaling vs encoding) must be applied.
        print("-" * 70)
        print("1. COLUMN METADATA & DATA TYPES")
        print("-" * 70)
        for col in df.columns:
            missing_cnt = df[col].isnull().sum()
            print(f"  • {col:<24} : Type={str(df[col].dtype):<10} | Null Count={missing_cnt}")

        # 3. Display Formatted Sample Records
        # WHY WRITTEN: Visual verification of the first 5 records ensures correct delimiter parsing.
        print_formatted_table(df, title="2. FIRST 5 SAMPLE RECORDS WITH SPACIOUS COLUMN GAPS")

        # 4. Analyze Missing and Duplicate Records
        # WHY WRITTEN: Duplicates cause data leakage and artificially inflate model accuracy;
        # missing values cause scikit-learn estimators to throw runtime errors.
        print("\n" + "-" * 70)
        print("3. MISSING & DUPLICATE VALUES SUMMARY")
        print("-" * 70)
        missing_sum = df.isnull().sum()
        cols_with_nulls = missing_sum[missing_sum > 0]
        if not cols_with_nulls.empty:
            print("  • Columns with Missing Values:")
            for c, cnt in cols_with_nulls.items():
                print(f"    - {c}: {cnt} missing ({round(cnt/len(df)*100, 2)}%)")
        else:
            print("  • No missing values detected across all columns!")
        
        print(f"\n  • Total Missing Cell Count : {df.isnull().sum().sum()}")
        print(f"  • Total Duplicate Records  : {df.duplicated().sum()}")

        # 5. Numerical Descriptive Statistical Overview
        # WHY WRITTEN: Computes count, mean, standard deviation, minimum, 25%, 50% (median),
        # 75%, and maximum for all continuous features to identify scale differences and potential outliers.
        print("\n" + "-" * 70)
        print("4. NUMERICAL DESCRIPTIVE STATISTICAL OVERVIEW")
        print("-" * 70)
        numeric_df = df.select_dtypes(include=np.number)
        if not numeric_df.empty:
            print(numeric_df.describe().round(2).to_string())
        else:
            print("  [NOTE] No numeric columns detected for descriptive statistics.")

        # 6. Generate CGPA Distribution Histogram
        # WHY WRITTEN: CGPA is a primary predictor for campus placements. Visualizing its frequency
        # distribution verifies normality and reveals student grade concentration.
        if "CGPA" in df.columns and pd.api.types.is_numeric_dtype(df["CGPA"]):
            plt.figure(figsize=(8, 5))
            plt.hist(df['CGPA'].dropna(), bins=15, color='#4A90E2', edgecolor='black', alpha=0.85)
            plt.title("CGPA Distribution Histogram (Campus Cohort)", fontsize=13, fontweight='bold')
            plt.xlabel("CGPA Score (0.0 - 10.0)", fontsize=11)
            plt.ylabel("Number of Students (Frequency)", fontsize=11)
            plt.grid(True, linestyle='--', alpha=0.6)
            
            hist_path = os.path.join(EDA_OUTPUT_DIR, "Histogram_CGPA.png")
            plt.tight_layout()
            plt.savefig(hist_path, dpi=300)
            plt.close()
            print(f"\n[INFO] CGPA Histogram chart saved to: {hist_path}")

        print("=" * 70)
        print("[SUCCESS] Module 1: Dataset Inspection & Metadata Analysis Completed.")
        print("=" * 70)

    except FileNotFoundError as fnf_err:
        print(f"[ERROR] File Not Found: {fnf_err}")
    except Exception as e:
        print(f"[ERROR] An unexpected error occurred during execution: {e}")


if __name__ == "__main__":
    main()

