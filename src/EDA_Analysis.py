import os
import io
import pandas as pd
import numpy as np
import matplotlib
# Headless Matplotlib Backend: Ensures reliable chart generation without requiring a GUI display server.
# WHY WRITTEN: When scripts run in backend services, cloud pipelines, or headless terminals, interactive GUI backends cause Fatal OS crashes.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# ====================================================================================================
# FUNCTION: load_dataset
# WHY WRITTEN:
# 1. Accommodates both standard CSV datasets and formatted ASCII/pipe-delimited boxed grid datasets.
# 2. Removes decorative row dividers (+-----+), trims whitespace from column names and string cells,
#    and casts numerical columns to appropriate numeric datatypes (float/int).
# ====================================================================================================
def load_dataset(filepath):
    """
    Safely load dataset supporting both standard comma-separated CSVs
    and human-readable boxed grid tables.
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
# MAIN SCRIPT EXECUTION
# WHY WRITTEN:
# Exploratory Data Analysis (EDA) is the foundational cornerstone of Machine Learning workflows.
# This script performs 5 comprehensive analytical stages:
#   1. Univariate Analysis: Evaluates the distribution, variance, spread, and outlier presence
#      for every continuous numerical feature using histograms and box plots.
#   2. Statistical Overview & Correlation Analysis: Computes summary metrics (mean, std, percentiles)
#      and calculates Pearson correlation coefficients between numeric features, saving both tabular
#      CSVs and a Seaborn heatmap to identify collinearity.
#   3. Categorical Distribution Analysis: Generates count plots for nominal/categorical attributes
#      (e.g., Department, Internship) to evaluate class frequencies and demographic balance.
#   4. Multivariate & Missingness Diagnostics: Generates pair plots to visualize feature interactions
#      and missing value heatmaps to pinpoint data sparsity.
#   5. Target Variable Class Imbalance Analysis: Visualizes the distribution of the dependent variable
#      (PlacementStatus: Placed vs. Not Placed) to inform classifier weighting and sampling decisions.
# ====================================================================================================
def main():
    # Setup Paths & Configurations
    # WHY WRITTEN: Using absolute paths based on __file__ guarantees platform-independent execution.
    BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    DATASET_PATH = os.path.join(BASE_DIR, "dataset", "placement_predict_50K_Raw.csv")
    OUTPUT_FOLDER = os.path.join(BASE_DIR, "outputs", "EDA_Analysis_outputs")
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    # Apply modern high-contrast aesthetic theme for published charts
    sns.set_theme(style="whitegrid")
    plt.rcParams["figure.figsize"] = (8, 5)

    print("=" * 70)
    print("             EXPLORATORY DATA ANALYSIS (EDA) PIPELINE")
    print("=" * 70)

    try:
        df = load_dataset(DATASET_PATH)

        print(f"[INFO] Dataset Loaded Successfully. Dimensions: {df.shape[0]} Rows x {df.shape[1]} Columns")
        print(f"[INFO] Total Null Cells: {df.isnull().sum().sum()} | Duplicate Records: {df.duplicated().sum()}")

        # --------------------------------------------------------------------------------------------
        # STEP 1: Feature Type Segregation & Target Variable Detection
        # WHY WRITTEN: Preprocessing, visualization styles, and encoding differ fundamentally between
        # numerical continuous variables (which require scaling/histograms) and categorical discrete
        # variables (which require frequency counts/encoding).
        # --------------------------------------------------------------------------------------------
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        numeric_df = df.select_dtypes(include=np.number)
        categorical_columns = df.select_dtypes(include=['object', 'string', 'category', 'bool']).columns.tolist()

        # Identify target candidate
        target = None
        for col in ["PlacementStatus", "Placement", "Status", "Placed"]:
            if col in df.columns:
                target = col
                break

        print(f"[INFO] Numerical Continuous Features ({len(numeric_cols)}): {', '.join(numeric_cols)}")
        print(f"[INFO] Categorical Discrete Features ({len(categorical_columns)}): {', '.join(categorical_columns)}")
        print(f"[INFO] Target Placement Attribute  : '{target}'")

        # --------------------------------------------------------------------------------------------
        # STEP 2: Univariate Analysis (Histograms with KDE & Box Plots)
        # WHY WRITTEN:
        # - Histograms with Kernel Density Estimation (KDE) show whether features are normally distributed,
        #   bimodal, or skewed. Skewed features may require logarithmic or power transformations.
        # - Box plots highlight interquartile ranges (IQR) and identify extreme outlier values that
        #   could distort linear classifiers or distance-based models (like KNN and Logistic Regression).
        # --------------------------------------------------------------------------------------------
        print("\n[INFO] 1. Generating Univariate Histograms & Box Plots...")
        for col in numeric_cols:
            # Histogram + KDE
            plt.figure(figsize=(8, 5))
            sns.histplot(df[col].dropna(), bins=20, kde=True, color='#3498db', edgecolor='black', alpha=0.7)
            plt.title(f"Univariate Distribution: {col}", fontsize=12, fontweight='bold')
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Frequency (Student Count)", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"Histogram_{col}.png"), dpi=150)
            plt.close()

            # Box Plot for Outliers
            plt.figure(figsize=(6, 4))
            sns.boxplot(y=df[col].dropna(), color="#e67e22")
            plt.title(f"Outlier & Dispersion Box Plot: {col}", fontsize=12, fontweight='bold')
            plt.ylabel(col, fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"Boxplot_{col}.png"), dpi=150)
            plt.close()

        # --------------------------------------------------------------------------------------------
        # STEP 3: Statistical Summary Export & Pearson Correlation Matrix
        # WHY WRITTEN:
        # - Statistical summaries (mean, std, 25%, 50%, 75%) provide a quantitative baseline.
        # - Correlation coefficients (ranging from -1.0 to +1.0) quantify linear associations between
        #   academic metrics (e.g., CGPA vs AptitudeScore) and assist in detecting multicollinearity.
        # --------------------------------------------------------------------------------------------
        print("[INFO] 2. Exporting Statistical Summaries & Correlation Matrix...")
        summary = df.describe(include='all')
        summary.to_csv(os.path.join(OUTPUT_FOLDER, "Statistical_Summary.csv"))

        if not numeric_df.empty and len(numeric_df.columns) > 1:
            corr = numeric_df.corr()
            corr.to_csv(os.path.join(OUTPUT_FOLDER, "Correlation_Matrix.csv"))

            plt.figure(figsize=(10, 8))
            sns.heatmap(corr, annot=True, cmap="coolwarm", fmt=".2f", linewidths=0.5, vmin=-1.0, vmax=1.0)
            plt.title("Pearson Correlation Coefficient Heatmap", fontsize=13, fontweight='bold')
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, "Correlation_Heatmap.png"), dpi=200)
            plt.close()

        # --------------------------------------------------------------------------------------------
        # STEP 4: Categorical Frequency Distributions (Count Plots)
        # WHY WRITTEN:
        # - Visualizing categorical column counts (e.g. Department, Internship) exposes class imbalance.
        # - Prevents models from developing bias toward majority branches (e.g. CSE) over minority branches.
        # --------------------------------------------------------------------------------------------
        print("[INFO] 3. Generating Categorical Frequency Distributions...")
        for col in categorical_columns:
            plt.figure(figsize=(8, 5))
            sns.countplot(data=df, x=col, hue=col, palette="viridis", legend=False)
            plt.xticks(rotation=25)
            plt.title(f"Frequency Distribution: {col}", fontsize=12, fontweight='bold')
            plt.xlabel(col, fontsize=10)
            plt.ylabel("Record Count", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, f"Countplot_{col}.png"), dpi=150)
            plt.close()

        # --------------------------------------------------------------------------------------------
        # STEP 5: Pairwise Feature Interactions & Missingness Matrix
        # WHY WRITTEN:
        # - Pair plots plot scatter plots for every combination of numeric features, revealing
        #   linear, quadratic, or clustered relationships.
        # - The missing values heatmap confirms dataset cleanliness before sending into ML pipelines.
        # --------------------------------------------------------------------------------------------
        print("[INFO] 4. Generating Pair Plot & Missingness Heatmap...")
        if len(numeric_df.columns) > 1 and len(numeric_df) > 2:
            pair = sns.pairplot(numeric_df.dropna())
            pair.savefig(os.path.join(OUTPUT_FOLDER, "Pairplot.png"))
            plt.close()

        plt.figure(figsize=(10, 6))
        sns.heatmap(df.isnull(), cbar=False, cmap="viridis")
        plt.title("Missing Values Matrix (Yellow = Missing, Purple = Clean)", fontsize=12, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(OUTPUT_FOLDER, "Missing_Values_Heatmap.png"), dpi=150)
        plt.close()

        # --------------------------------------------------------------------------------------------
        # STEP 6: Target Outcome Distribution (Placed vs. Not Placed)
        # WHY WRITTEN:
        # - Imbalanced target classes (e.g. 95% Placed, 5% Not Placed) lead to accuracy paradoxes.
        # - Confirming the class balance helps choose stratified sampling in train_test_split.
        # --------------------------------------------------------------------------------------------
        if target:
            print(f"[INFO] 5. Visualizing Target Distribution for '{target}'...")
            plt.figure(figsize=(6, 4))
            sns.countplot(data=df, x=target, hue=target, palette="Set2", legend=False)
            plt.title(f"Target Distribution: {target} (Placed vs. Not Placed)", fontsize=12, fontweight='bold')
            plt.xlabel("Placement Status", fontsize=10)
            plt.ylabel("Student Count", fontsize=10)
            plt.tight_layout()
            plt.savefig(os.path.join(OUTPUT_FOLDER, "Target_Distribution.png"), dpi=150)
            plt.close()

        print("\n" + "=" * 70)
        print("                  SUMMARY OF EDA ARTIFACTS GENERATED")
        print("=" * 70)
        print(f"  • Destination Folder               : {OUTPUT_FOLDER}")
        print(f"  • Summary Statistics Table Saved  : Statistical_Summary.csv")
        print(f"  • Pearson Correlation Table Saved  : Correlation_Matrix.csv")
        print(f"  • Correlation Matrix Heatmap       : Correlation_Heatmap.png")
        print(f"  • Missing Values Matrix Heatmap    : Missing_Values_Heatmap.png")
        print(f"  • Univariate Histograms & Boxplots : Histogram_*.png, Boxplot_*.png")
        print("=" * 70)
        print("[SUCCESS] Exploratory Data Analysis Pipeline Completed Successfully.")
        print("=" * 70)

    except FileNotFoundError as fnf:
        print(f"[ERROR] {fnf}")
    except Exception as ex:
        print(f"[ERROR] An unexpected error occurred during EDA execution: {ex}")


if __name__ == "__main__":
    main()

