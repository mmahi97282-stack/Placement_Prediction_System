import os
import io
import pandas as pd
import numpy as np
import matplotlib
# Headless Matplotlib Backend: Prevents GUI window errors during automated background execution
# WHY WRITTEN: Server-side data processing requires generating charts directly to PNG files without an interactive display.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Supports reading both raw comma-separated CSVs and pipe-delimited ASCII boxed grid tables.
# 2. Strips whitespace, removes decorative border artifacts (+----+), and casts data cleanly.
# ====================================================================================================
def load_dataset(filepath):
    """
    Safely load dataset supporting standard CSVs and human-readable boxed grid tables.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"[ERROR] Dataset file not found at: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if pipe grid table
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
# FUNCTION: print_formatted_table
# WHY WRITTEN:
# 1. Renders clean ASCII boxed tables to the terminal with customized column widths.
# 2. Helps engineers immediately inspect column alignments, data representations, and padding.
# ====================================================================================================
def print_formatted_table(df_sample, title="Dataset Sample", max_rows=5, padding=4):
    """
    Renders an ASCII boxed grid table for terminal output with dynamic column spacing.
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
# MAIN SCRIPT EXECUTION
# WHY WRITTEN:
# Module 1 (Missing Value & Duplicate Identification Pipeline) is designed to:
#   1. Detect all null, NaN, None, and empty records across both numerical and categorical features.
#   2. Compute percentage missingness to determine if feature imputation (mean/mode) or column deletion is warranted.
#   3. Scan for duplicate student entries which could contaminate the train/test split.
#   4. Generate a seaborn missingness heatmap saved to disk for visual inspection.
# ====================================================================================================
def main():
    # Setup Paths & Configurations
    # WHY WRITTEN: Using absolute paths based on __file__ prevents path resolution bugs when scripts are executed from different subdirectories.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    EDA_OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "EDA_Analysis_outputs")
    os.makedirs(EDA_OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print(" MODULE 1: IDENTIFY MISSING VALUES & DUPLICATES IN DATASET")
    print("=" * 70)

    try:
        # Load dataset cleanly
        df = load_dataset(DATASET_PATH)

        print(f"\n[INFO] Dataset Successfully Loaded from: {DATASET_PATH}")
        print(f"[INFO] Dataset Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns\n")

        # Step 1: Display Top Sample Records
        print_formatted_table(df, title="1. RAW DATASET PREVIEW (FIRST 5 ROWS)")

        # Step 2: Calculate Missing Values per Column
        # WHY WRITTEN: Scikit-learn classification algorithms cannot natively process missing values.
        # Quantifying missingness per column allows selecting appropriate imputation strategies (mean for numeric, mode for categorical).
        print("\n" + "-" * 70)
        print("2. MISSING VALUE QUANTIFICATION PER FEATURE")
        print("-" * 70)
        missing_counts = df.isnull().sum()
        for col in df.columns:
            cnt = missing_counts[col]
            pct = round((cnt / len(df)) * 100, 2)
            status = "CLEAN (0% Missing)" if cnt == 0 else f"ACTION REQUIRED ({cnt} Missing, {pct}%)"
            print(f"  • {col:<24} : {cnt:<4} missing values | {status}")

        total_missing = missing_counts.sum()
        print(f"\n  • Total Missing Cells in Dataset : {total_missing}")

        # Step 3: Duplicate Record Detection
        # WHY WRITTEN: Duplicate records cause data leakage between training and testing sets,
        # leading to over-optimistic accuracy metrics that fail in production.
        print("\n" + "-" * 70)
        print("3. DUPLICATE STUDENT RECORDS ANALYSIS")
        print("-" * 70)
        duplicate_rows = df[df.duplicated()]
        num_duplicates = len(duplicate_rows)
        print(f"  • Duplicate records detected : {num_duplicates}")
        if num_duplicates > 0:
            print("  • Note: Module 2 Preprocessing pipeline will drop duplicate rows automatically.")
        else:
            print("  • All rows in dataset represent unique student records.")

        # Step 4: Missingness Heatmap Visualization
        # WHY WRITTEN: A 2D heatmap matrix visually highlights whether missing values occur randomly
        # or follow systematic patterns across specific batches of students.
        print("\n" + "-" * 70)
        print("4. GENERATING MISSINGNESS HEATMAP VISUALIZATION")
        print("-" * 70)
        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, yticklabels=False, cmap="viridis")
        plt.title("Missing Values Distribution Heatmap (Yellow = Missing, Purple = Present)", fontsize=12, fontweight='bold')
        plt.xlabel("Dataset Features", fontsize=10)
        plt.ylabel("Student Records", fontsize=10)
        heatmap_path = os.path.join(EDA_OUTPUT_DIR, "Missing_Values_Heatmap.png")
        plt.tight_layout()
        plt.savefig(heatmap_path, dpi=300)
        plt.close()
        print(f"  [INFO] Missingness Heatmap saved successfully to: {heatmap_path}")

        print("=" * 70)
        print("[SUCCESS] Module 1: Missing Value & Duplicate Identification Completed.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

