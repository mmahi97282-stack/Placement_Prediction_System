import os
import io
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

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
# MAIN SCRIPT EXECUTION: DATASET FEATURE & SUBSET SEPARATION PIPELINE
# WHY WRITTEN:
# In professional Machine Learning architectures, raw monolith datasets are split into modular,
# decoupled subsets for auditing, reproducibility, and dedicated transformer pipelines:
#   1. Numerical Features Subset (`numerical_features.csv`):
#      - Isolates continuous interval metrics (CGPA, AptitudeScore, CommunicationScore, Backlogs).
#      - Enables dedicated statistical scaling pipelines (StandardScaler, MinMaxScaler) without touching text columns.
#   2. Categorical Features Subset (`categorical_features.csv`):
#      - Isolates discrete nominal/ordinal attributes (Department, Internship).
#      - Enables categorical encoders (OneHotEncoder, LabelEncoder) to run in isolation.
#   3. Target Variable Subset (`target_variable.csv`):
#      - Isolates ground-truth outcomes ('PlacementStatus') along with student identifiers for auditing.
#   4. Predictor Matrix X (`X_predictors.csv`) & Target Vector y (`y_target.csv`):
#      - Strictly separates the independent feature space (X) from the dependent target labels (y).
#   5. Train/Test Split (`train_dataset.csv` & `test_dataset.csv`):
#      - Implements an 80/20 train/test split with fixed random_state=42.
#      - Guarantees unbiased model validation without data leakage between training and testing splits.
# ====================================================================================================
# ====================================================================================================
# FUNCTION: save_as_boxed_grid_table
# WHY WRITTEN:
# Serializes subsets as human-readable ASCII boxed grid tables with pipe boundaries.
# ====================================================================================================
def save_as_boxed_grid_table(df, filepath):
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


def main():
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_DIR = os.path.join(BASE_DIR, "outputs", "separated_data")

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("       DATASET FEATURE & SUBSET SEPARATION PIPELINE")
    print("=" * 70)

    try:
        df = load_dataset(DATASET_PATH)

        print(f"[INFO] Raw Dataset Successfully Loaded: {df.shape[0]} Rows x {df.shape[1]} Columns\n")

        # Step 1: Detect Target Column
        TARGET_CANDIDATES = ["PlacementStatus", "Placement", "Status", "Placed"]
        target_column = next((c for c in TARGET_CANDIDATES if c in df.columns), df.columns[-1])

        id_col = "StudentID" if "StudentID" in df.columns else None

        num_cols = df.select_dtypes(include=np.number).columns.tolist()
        cat_cols = df.select_dtypes(exclude=np.number).columns.tolist()

        # Step 2: Extract Numerical Features Subset
        num_features_df = df[num_cols].copy()

        # Step 3: Extract Categorical Features Subset
        cat_features_cols = cat_cols.copy()
        if id_col and id_col not in cat_features_cols and id_col in df.columns:
            cat_features_cols.insert(0, id_col)
        cat_features_df = df[cat_features_cols].copy()

        # Step 4: Extract Target Variable Subset
        target_cols = [col for col in [id_col, target_column] if col and col in df.columns]
        target_df = df[target_cols].copy()

        # Step 5: Predictors (X) and Target (y) Matrices
        X_df = df.drop(columns=[target_column]) if target_column in df.columns else df.copy()
        y_df = df[[target_column]] if target_column in df.columns else pd.DataFrame()

        # Step 6: 80/20 Train / Test Stratified Split
        # WHY WRITTEN: 80% of data is used for model parameter learning; 20% is withheld for unbiased generalization evaluation.
        if not y_df.empty and len(df) > 4:
            X_train, X_test, y_train, y_test = train_test_split(
                X_df, y_df, test_size=0.2, random_state=42
            )
            train_df = pd.concat([X_train, y_train], axis=1)
            test_df = pd.concat([X_test, y_test], axis=1)
        else:
            train_df, test_df = df.copy(), pd.DataFrame()

        # Step 7: Export Subsets to outputs/separated_data/
        subsets = {
            "numerical_features.csv": num_features_df,
            "categorical_features.csv": cat_features_df,
            "target_variable.csv": target_df,
            "X_predictors.csv": X_df,
            "y_target.csv": y_df,
            "train_dataset.csv": train_df,
            "test_dataset.csv": test_df,
        }

        print("-" * 70)
        print("            SEPARATED SUBSET DATASET SUMMARY")
        print("-" * 70)

        for filename, subset_data in subsets.items():
            if not subset_data.empty:
                out_path = os.path.join(OUTPUT_DIR, filename)
                save_as_boxed_grid_table(subset_data, out_path)
                print(f"  • {filename:<25} : {subset_data.shape[0]} rows x {subset_data.shape[1]} cols (Boxed Grid Table)")
                print(f"    Features: {', '.join(subset_data.columns.tolist()[:8])}\n")

        print("=" * 70)
        print("[SUCCESS] All Modular Dataset Subsets Saved in Boxed Grid Table Format Into outputs/separated_data.")
        print("=" * 70)

    except FileNotFoundError as fnf:

        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred: {ex}")


if __name__ == "__main__":
    main()

