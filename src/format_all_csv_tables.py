import os
import glob
import io
import pandas as pd

# ====================================================================================================
# CONFIGURATION & DIRECTORY PATHS
# WHY WRITTEN:
# Sets up directory targets across both dataset/ and outputs/cleaned_data/ to format all CSV files.
# ====================================================================================================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATASET_DIR = os.path.join(BASE_DIR, "dataset")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")


# ====================================================================================================
# FUNCTION: format_csv_as_table
# WHY WRITTEN:
# 1. Aligns all cells in a CSV file with dynamic column padding so that opening the file in Notepad,
#    VS Code, or cat in the terminal presents a clean, readable text table without unaligned commas.
# 2. Cleans whitespace from column headers and cell values.
# ====================================================================================================
def format_csv_as_table(filepath):
    """
    Reads a CSV, strips excess whitespace, and rewrites it with aligned column widths.
    """
    if not os.path.exists(filepath):
        return
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            content = f.read()
        
        if "|" in content:
            lines = [l for l in content.splitlines() if not l.strip().startswith("+") and l.strip()]
            df = pd.read_csv(io.StringIO("\n".join(lines)), sep="|", skipinitialspace=True).dropna(how="all", axis=1)
        else:
            df = pd.read_csv(filepath, skipinitialspace=True)
            
        for col in df.columns:
            if pd.api.types.is_string_dtype(df[col]) or pd.api.types.is_object_dtype(df[col]):
                df[col] = df[col].astype(str).str.strip()
            df.rename(columns={col: str(col).strip()}, inplace=True)
            
        cols = [str(c) for c in df.columns]
        widths = {c: max([len(str(val)) for val in df[c]] + [len(c)]) for c in cols}
        
        # Build aligned CSV lines
        header_line = " , ".join([f"{c:<{widths[c]}}" for c in cols])
        data_rows = [header_line]
        
        for _, row in df.iterrows():
            row_line = " , ".join([f"{str(row[c]):<{widths[c]}}" for c in cols])
            data_rows.append(row_line)
            
        formatted_content = "\n".join(data_rows) + "\n"
        
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(formatted_content)
            
        print(f"  • [SUCCESS] Aligned CSV Table Saved: {os.path.basename(filepath)}")
    except Exception as e:
        print(f"  • [ERROR] Failed to format {filepath}: {e}")


# ====================================================================================================
# MAIN SCRIPT EXECUTION
# WHY WRITTEN:
# Formats all CSV datasets in both dataset/ and outputs/ folders.
# ====================================================================================================
def main():
    print("=" * 70)
    print("      FORMATTING ALL CSV DATASETS AS ALIGNED TABLE FILES")
    print("=" * 70)
    
    for dir_path in [DATASET_DIR, OUTPUT_DIR]:
        if os.path.exists(dir_path):
            for file in glob.glob(os.path.join(dir_path, "*.csv")):
                format_csv_as_table(file)

    print("=" * 70)
    print("[SUCCESS] All CSV Files Formatted with Uniform Spacing.")
    print("=" * 70)


if __name__ == "__main__":
    main()

