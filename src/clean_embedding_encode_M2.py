import os
import io
import pandas as pd
import numpy as np

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Accommodates standard comma-separated CSVs and human-readable boxed grid ASCII/pipe tables.
# 2. Ensures data types and whitespace are cleaned before applying embedding projections.
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
    widths = {c: max([len(str(val)) for val in sample[c]] + [len(c)]) for c in cols}
    
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
# MAIN SCRIPT EXECUTION: MODULE 2 (CLEANING & EMBEDDING VECTOR ENCODING)
# WHY WRITTEN:
# Embedding Vector Encoding projects discrete categorical categories into a continuous d-dimensional coordinate space:
#   1. Continuous Latent Representation: Mimics deep learning embedding layers (like Word2Vec, PyTorch nn.Embedding),
#      representing each category as a vector of real numbers [v1, v2, v3].
#   2. Geometric Distance Capability: Dense embedding vectors allow calculating Euclidean distances or Cosine
#      similarities between student department profiles in high-dimensional feature spaces.
#   3. Dimensionality Compression: Avoids sparse wide matrices when categories grow to hundreds of unique entries.
# ====================================================================================================
def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "cleaned_data")
    DATASET_DIR = os.path.join(BASE_DIR, "dataset")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(DATASET_DIR, exist_ok=True)

    print("=" * 70)
    print("    MODULE 2: CLEANING & EMBEDDING ENCODING PIPELINE")
    print("=" * 70)

    try:
        data = load_dataset(DATASET_PATH)

        print(f"[INFO] Raw Input Dataset Loaded. Dimensions: {data.shape[0]} Rows x {data.shape[1]} Columns")
        print_formatted_table(data, title="Raw Input Dataset Sample (First 5 Rows)")

        # Step 1: Whitespace Normalization
        for col in data.select_dtypes(include=["object", "string"]).columns:
            data[col] = data[col].astype(str).str.strip()

        # Step 2: Duplicate Check and Cleaning
        duplicate_count = data.duplicated().sum()
        data = data.drop_duplicates()
        print(f"\n[INFO] Duplicate Records Removed: {duplicate_count}")

        # Step 3: Numerical and Categorical Separation
        num_cols = data.select_dtypes(include=np.number).columns.tolist()
        cat_cols = data.select_dtypes(exclude=np.number).columns.tolist()

        # Step 4: Missing Value Imputation
        for col in num_cols:
            data[col] = data[col].fillna(data[col].mean())

        for col in cat_cols:
            data[col] = data[col].fillna(data[col].mode()[0])

        # Step 5: Dense Coordinate Vector (Embedding) Construction
        # WHY WRITTEN: Maps each category to a 3-dimensional unit vector in continuous space.
        embedding_output = pd.DataFrame()
        embedding_size = 3

        for col in cat_cols:
            categories = data[col].unique()
            embedding_matrix = {}

            for index, category in enumerate(categories):
                vector = np.zeros(embedding_size)
                vector[index % embedding_size] = 1.0
                embedding_matrix[category] = vector

            embeddings = data[col].map(embedding_matrix)

            embedding_df = pd.DataFrame(
                embeddings.tolist(),
                columns=[
                    f"Embedding_{col}_1",
                    f"Embedding_{col}_2",
                    f"Embedding_{col}_3"
                ]
            )

            embedding_output = pd.concat([embedding_output, embedding_df], axis=1)

        final_output = pd.concat(
            [data[num_cols].reset_index(drop=True), embedding_output.reset_index(drop=True)],
            axis=1
        )

        total_missing_after = final_output.isnull().sum().sum()

        # Step 6: Dual Storage Output Persistence
        output_file_outputs = os.path.join(OUTPUT_DIR, "clean_embedded_encode_M2.csv")
        output_file_dataset = os.path.join(DATASET_DIR, "clean_embedded_encode_M2.csv")

        save_table_csv(final_output, output_file_outputs)
        save_table_csv(final_output, output_file_dataset)

        print_formatted_table(final_output.iloc[:, :10], title="Cleaned & Embedding Encoded Dataset (First 10 Features)")

        print("\n" + "-" * 70)
        print("                  SUMMARY OF EMBEDDING ENCODED DATASET")
        print("-" * 70)
        print(f"  • Final Output Dataset Shape : {final_output.shape[0]} Rows x {final_output.shape[1]} Columns")
        print(f"  • Embedding Dimension        : {embedding_size} Dimensions per Categorical Column")
        print(f"  • Primary Output Saved To    : {output_file_outputs}")
        print(f"  • Dataset Folder Sync To     : {output_file_dataset}")
        print("=" * 70)
        print("[SUCCESS] Module 2: Embedding Encoding Pipeline Completed Successfully.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()
