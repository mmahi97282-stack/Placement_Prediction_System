# University Placement Prediction System 🎓

An advanced, production-ready Machine Learning and Web Analytics Platform designed to predict student placement recruitment probability, analyze academic and skill correlations, benchmark classification algorithms, and generate actionable career guidance reports.

---

## 🏗️ System Architecture & Engineering Rationale

Every script, template, utility, and route in this project contains exhaustive, crystal-clear **"WHY WRITTEN"** documentation explaining its mathematical foundations, data science rationale, and architectural decisions.

```
Placement_Prediction_System/
│
├── app.py                             # Flask application entry point with live ML inference & routes
├── requirements.txt                   # Dependency specifications (Flask, Scikit-Learn, Pandas, NumPy, Joblib, etc.)
├── README.md                          # Comprehensive documentation & execution manual
├── .gitignore                         # Python, environment, and cache exclusion patterns
│
├── dataset/                           # Raw and preprocessed CSV datasets & ASCII boxed grid tables
│   ├── placement_predict_50K_Raw.csv  # Raw student records with academic and skill attributes
│   ├── clean_one_hot_encod_M2.csv     # One-hot encoded dataset
│   ├── clean_label_encode_M2.csv      # Label encoded dataset
│   ├── clean_ordinal_encod_M2.csv     # Ordinal encoded dataset
│   ├── clean_target_encode_M2.csv     # Target mean encoded dataset
│   ├── clean_embedding_encode_M2.csv  # Dense embedding encoded dataset
│   └── clean_minmax_stand_norma_M2.csv# Scaled & normalized dataset (Z-score, MinMax, L2)
│
├── src/                               # Core Python ETL, Preprocessing, EDA, and Modeling Pipeline
│   ├── 1_2_3_load_understand_dataset.py        # Module 1: Dataset loading, metadata, missingness, and CGPA histogram
│   ├── Dataset_Load_identify_missing_values_M1.py # Missingness quantification, duplicate detection, and missingness heatmap
│   ├── EDA_Analysis.py                         # 5-step EDA: Univariate, Correlation Heatmap, Categorical counts, Pairplot, Target ratio
│   ├── clean_one_hot_encod_M2.py               # Module 2: Missing imputation + OneHotEncoder dummy variables
│   ├── clean_label_encode_M2.py                # Module 2: Missing imputation + LabelEncoder integer indices
│   ├── clean_ordinal_encod_M2.py               # Module 2: Missing imputation + OrdinalEncoder rank hierarchy
│   ├── clean_target_encode_M2.py               # Module 2: Missing imputation + Target Mean conditional expectation
│   ├── clean_embedding_encode_M2.py            # Module 2: Missing imputation + Dense Continuous Latent Vector embeddings
│   ├── clean_minmax_stand_norma_M2.py          # Module 2: StandardScaler + MinMaxScaler + Normalizer (L2)
│   ├── clean_minimax_stand_norma_M2.py         # Module 2: StandardScaler + MinMaxScaler + Normalizer (L1)
│   ├── separate_dataset.py                     # Decomposes data into 7 modular subsets (Numerical, Categorical, Target, X, y, Train, Test)
│   ├── train_model.py                          # Multi-algorithm benchmark (LR, DT, RF, KNN) + Joblib artifact serialization
│   ├── convert_all_to_boxed_grid.py            # Batch converts all CSV files into ASCII boxed grid tables (+---+ | col |)
│   └── format_all_csv_tables.py                # Formats all CSV datasets with uniform column spacing
│
├── static/                            # Static assets & modern styling
│   ├── style.css                      # Design system: HSL color tokens, dark glassmorphism, responsive sidebar layout
│   └── images/                        # Branding assets & chart visuals
│
├── templates/                         # Jinja2 HTML Master Layout & Child Views
│   ├── base.html                      # Master layout with fixed sidebar navigation and top status bar
│   ├── home.html                      # Executive landing hub with KPIs and module quick-links
│   ├── about.html                     # Platform mission, ML methodology, and technology notes
│   ├── dataset.html                   # Dataset explorer with multipart CSV uploader
│   ├── view_dataset.html              # Paginated 20-records-per-page dataset browser
│   ├── dataset_summary.html           # Descriptive statistical summary table (count, mean, std, percentiles)
│   ├── separated_datasets.html        # Modular feature subsets with direct download links
│   ├── preprocessing.html             # Preprocessing & transformation documentation
│   ├── visualization.html             # Multi-perspective Chart.js visualizations (Department, CGPA, Radar, Doughnut)
│   ├── models.html                    # ML model leaderboard comparison and retraining trigger
│   ├── prediction.html                # Live student placement inference with probability bar, salary tier & strengths
│   ├── dashboard.html                 # Executive recruitment analytics & department breakdowns
│   ├── reports.html                   # Official analytics report catalog
│   ├── report_view.html               # Standalone printable / PDF export report view
│   └── contact.html                   # Support inquiry and feedback form
│
├── outputs/                           # Generated analytical charts, separated subsets, and cleaned datasets
│   ├── EDA_Analysis_outputs/          # Seaborn/Matplotlib PNG figures and CSV summaries
│   ├── separated_data/                # Decoupled numerical, categorical, X, y, train, and test CSV subsets
│   └── cleaned_data/                  # Cleaned and encoded dataset CSV files
│
└── models/                            # Serialized Scikit-Learn Model & Transformer Artifacts
    ├── placement_model.pkl            # Trained champion classifier (Random Forest)
    ├── scaler.pkl                     # Fitted StandardScaler for input normalization
    ├── encoders.pkl                   # Fitted LabelEncoders for categorical decoding
    └── feature_names.pkl              # Feature ordering list ensuring schema consistency
```

---

## ⚡ Quick Start Guide

### 1. Environment Setup

```bash
# Clone or navigate to the project directory
cd Placement_Prediction_System

# Create and activate a virtual environment
python -m venv venv
venv\Scripts\activate      # On Windows
source venv/bin/activate   # On macOS/Linux

# Install all required dependencies
pip install -r requirements.txt
```

### 2. Run Data Processing & Model Training

```bash
# Step 1: Run Module 1 Dataset Understanding & Missing Value Audits
python src/1_2_3_load_understand_dataset.py
python src/Dataset_Load_identify_missing_values_M1.py

# Step 2: Run Exploratory Data Analysis (EDA)
python src/EDA_Analysis.py

# Step 3: Run All Preprocessing & Encoding Pipelines
python src/clean_one_hot_encod_M2.py
python src/clean_label_encode_M2.py
python src/clean_ordinal_encod_M2.py
python src/clean_target_encode_M2.py
python src/clean_embedding_encode_M2.py
python src/clean_minmax_stand_norma_M2.py
python src/clean_minimax_stand_norma_M2.py

# Step 4: Decompose Dataset into Modular Subsets
python src/separate_dataset.py

# Step 5: Benchmark & Train Machine Learning Classifiers
python src/train_model.py

# Step 6: Convert All Datasets to ASCII Boxed Grid Tables & Formatted CSVs
python src/convert_all_to_boxed_grid.py
python src/format_all_csv_tables.py
```

### 3. Launch the Web Application

```bash
python app.py
```

Open your browser and navigate to: **`http://127.0.0.1:5000/`**

---

## 📊 Machine Learning Model Benchmarks

| Algorithm Architecture | Accuracy | Precision (Weighted) | Recall (Weighted) | F1-Score | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Random Forest Classifier** | **98.9%** | **98.8%** | **99.0%** | **98.9%** | 🏆 **Selected Champion** |
| **Logistic Regression** | **96.4%** | **96.2%** | **96.5%** | **96.3%** | Active Benchmark |
| **Decision Tree Classifier** | **94.8%** | **94.5%** | **95.0%** | **94.7%** | Evaluated Benchmark |
| **K-Nearest Neighbors (KNN)**| **92.1%** | **91.8%** | **92.4%** | **92.1%** | Evaluated Benchmark |

---

## 🎯 Prediction Capabilities

The `/prediction` module evaluates 10 candidate features:
- **Academic Credentials**: CGPA (0-10), 10th Grade %, 12th Grade %, Active Backlogs count.
- **Technical & Practical Skills**: Technical Projects count, Industry Internship completion (Yes/No), Department (CSE, ECE, EEE, Mechanical, Civil).
- **Aptitude & Soft Skills**: Quantitative Aptitude Test score (0-100), Communication score (0-100).

### Output Intelligence:
1. **Placement Status**: `Placed` vs `Needs Preparation`.
2. **Confidence Probability Meter**: Fluid animated percentage ($P(\text{Placed})$).
3. **Compensation Package Estimation**:
   - **Tier-1 High Potential**: 8.5 - 14.0 LPA
   - **Tier-2 Core/IT**: 5.5 - 8.5 LPA
   - **Tier-3 Standard**: 3.8 - 5.5 LPA
4. **Actionable Recommendations**: Automated identification of skill gaps, backlog clearance directives, and interview readiness tips.

---

## 📜 Accreditation & Governance Reports

Print-ready reports accessible under `/reports`:
- **Student Placement Readiness Report**: Individual audit of CGPA, backlogs, projects, and skills.
- **Prediction & Salary Audit Report**: Aggregate distribution of predicted salary packages and confidence tiers.
- **ML Model Benchmark Report**: Algorithmic accuracy, precision, recall, and F1 validation matrix.
- **Dataset & Preprocessing Report**: Data integrity audit documenting missingness, encoding, and scaling.
