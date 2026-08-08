import os
import glob
import io
import pandas as pd

# ====================================================================================================
# CONFIGURATION & PATH SETUP
# WHY WRITTEN:
# 1. Automates the batch transformation of all dataset CSV files across both raw dataset repositories
#    and cleaned preprocessed output directories into human-readable ASCII boxed grid tables.
# 2. Ensures unified relative path management regardless of current working directory.
# ====================================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
SEPARATED_DIR = os.path.join(BASE_DIR, "outputs", "separated_data")


# ====================================================================================================
# FUNCTION: load_any_dataset
# WHY WRITTEN:
# 1. Accommodates standard comma-separated CSVs and pre-existing pipe-delimited boxed grid tables.
# 2. Strips surrounding border lines, removes empty divider rows, and trims text cell padding.
# ====================================================================================================
def load_any_dataset(filepath):
    """
    Safely load CSV or boxed grid table dataset into a clean pandas DataFrame.
    """
    with open(filepath, "r", encoding="utf-8") as f:
        content = f.read()
    
    # Check if pipe-delimited ASCII grid table
    if "|" in content:
        lines = [l for l in content.splitlines() if not l.strip().startswith("+") and l.strip()]
        df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", skipinitialspace=True).dropna(how="all", axis=1)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
        return df
    else:
        df = pd.read_csv(filepath, skipinitialspace=True)
        df.columns = [c.strip() for c in df.columns]
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
        return df


# ====================================================================================================
# FUNCTION: save_as_boxed_grid_table
# WHY WRITTEN:
# 1. Converts flat tabular data into professional, human-readable ASCII boxed grid tables.
# 2. Dynamically calculates individual column widths based on maximum value length + 2 spaces padding.
# 3. Employs ASCII border rows (+-------+-------+) and pipe separators (| value |) for clear visual boundaries.
# ====================================================================================================
def save_as_boxed_grid_table(df, filepath):
    """
    Serializes a pandas DataFrame as a fully boxed ASCII grid table with pipe delimiters.
    """
    cols = [str(c) for c in df.columns]
    widths = {c: max([len(str(val)) for val in df[c]] + [len(c)]) + 2 for c in cols}
    
    border = "+" + "+".join(["-" * widths[c] for c in cols]) + "+"
    header = "|" + "|".join([f" {c:<{widths[c]-1}}" for c in cols]) + "|"
    
    lines = [border, header, border]
    for _, row in df.iterrows():
        row_str = "|" + "|".join([f" {str(row[c]):<{widths[c]-1}}" for c in cols]) + "|"
        lines.append(row_str)
    lines.append(border)
    
    formatted_content = "\n".join(lines) + "\n"
    
    with open(filepath, "w", encoding="utf-8") as f:
        f.write(formatted_content)


# ====================================================================================================
# MAIN SCRIPT EXECUTION
# WHY WRITTEN:
# Scans all .csv files in dataset/, outputs/cleaned_data/, and outputs/separated_data/ and applies uniform boxed grid formatting.
# ====================================================================================================
def main():
    print("=" * 70)
    print("      CONVERTING ALL DATASETS TO FULL BOXED GRID TABLES")
    print("=" * 70)
    
    for dir_path in [DATASET_DIR, OUTPUT_DIR, SEPARATED_DIR]:
        if os.path.exists(dir_path):
            for file in glob.glob(os.path.join(dir_path, "*.csv")):
                try:
                    df = load_any_dataset(file)
                    save_as_boxed_grid_table(df, file)
                    print(f"  • [SUCCESS] Boxed Grid Table Saved: {os.path.basename(file)}")
                except Exception as e:
                    print(f"  • [ERROR] Failed to convert {file}: {e}")
                    
    print("=" * 70)
    print("[SUCCESS] All CSV Files Converted to Boxed Grid Tables.")
    print("=" * 70)


if __name__ == "__main__":
    main()


