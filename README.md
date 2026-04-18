# 🚗 Car Price Prediction

A machine learning project that predicts the **selling price of used cars** based on features like age, mileage, fuel type, and transmission. Built as a term project using a real-world dataset from CarDekho (Kaggle), comparing three regression models to find the best performer.

---

## 📌 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Project Structure](#project-structure)
- [ML Pipeline](#ml-pipeline)
- [Models Used](#models-used)
- [Results](#results)
- [How to Run](#how-to-run)
- [Requirements](#requirements)
- [Releases](#releases)
- [Packages](#packages)
- [Team](#team)

---

## Overview

Used car prices are influenced by many factors — how old the car is, how many kilometres it has been driven, what fuel it runs on, and whether it was sold by a dealer or a private individual. This project builds a complete end-to-end ML pipeline that:

1. Loads and explores the raw dataset
2. Cleans the data and engineers useful features (e.g. `Car_Age` from `Year`)
3. Encodes categorical variables (fuel type, seller type, transmission, owner)
4. Trains and evaluates three regression models
5. Lets you input your own car details and get a predicted price

The project is contained in a single Jupyter notebook (`rgressions.ipynb`) with a clean CSV dataset (`car_data.csv`).

---

## Dataset

**Source:** [Car Price Prediction — Kaggle](https://www.kaggle.com/datasets/zafarali27/car-price-prediction/data) by Zafarali27

| Column | Type | Description |
|---|---|---|
| `Car_Name` | string | Model name of the car (e.g. *ritz*, *swift*) |
| `Year` | int | Year the car was manufactured |
| `Selling_Price` | float | Listed selling price in **lakhs (₹)** — **target variable** |
| `Present_Price` | float | Current showroom price of the car |
| `Kms_Driven` | int | Total kilometres driven |
| `Fuel_Type` | string | `Petrol` / `Diesel` / `CNG` |
| `Seller_Type` | string | `Dealer` / `Individual` |
| `Transmission` | string | `Manual` / `Automatic` |
| `Owner` | int | Number of previous owners (0 = first owner) |

**Dataset stats:**
- 301 rows × 9 columns
- Price range: ₹0.10 lakhs – ₹35.00 lakhs
- No missing values
- Fuel types: Petrol, Diesel, CNG
- Seller types: Dealer, Individual
- Transmissions: Manual, Automatic

---

## Project Structure

```
Car-Price-Prediction/
│
├── car_data.csv          # Raw dataset (301 used car listings)
├── rgressions.ipynb      # Full ML pipeline notebook
└── README.md             # This file
```

---

## ML Pipeline

The notebook is structured as a step-by-step pipeline:

### Step 1 — Import Libraries
All dependencies are imported up front: `pandas`, `numpy`, `matplotlib`, `seaborn`, `scikit-learn`, and `xgboost`.

### Step 2 — Load Dataset
The CSV is loaded with `pandas`. The notebook supports both local file paths and Google Colab's interactive file upload.

### Step 3 — Exploratory Data Analysis (EDA)
- Dataset shape, dtypes, and non-null counts
- Statistical summary (min, max, mean, quartiles)
- Missing value check and duplicate row detection
- Distribution of the target variable (`Selling_Price`) — raw and log-transformed
- Average price by categorical features (fuel type, seller type, transmission)
- Scatter plots: `Year` vs price, `Kms_Driven` vs price
- Pearson correlation heatmap for numerical features

Key findings from EDA:
- Price distribution is heavily right-skewed (many cheap cars, few expensive ones)
- Newer cars and automatic transmission cars fetch higher prices
- Diesel cars tend to be priced higher than petrol cars
- `Year` has moderate positive correlation (~0.42) with selling price
- `Kms_Driven` has a weak negative correlation (~−0.23)

### Step 4 — Data Cleaning & Preprocessing
- **Duplicate rows removed** to prevent model bias
- **`Car_Name` dropped** — too many unique values (~300+) to encode meaningfully; would cause overfitting
- **`Car_Age` feature created** — derived as `current_year − Year`, then `Year` is dropped. Age is a more direct signal of depreciation than the raw year

### Step 5 — Feature Engineering & Encoding
Categorical columns are encoded using `LabelEncoder` from scikit-learn:

| Column | Encoding |
|---|---|
| `Fuel_Type` | Label encoded (Petrol=2, Diesel=0, CNG=1) |
| `Seller_Type` | Label encoded (Dealer=0, Individual=1) |
| `Transmission` | Label encoded (Manual=1, Automatic=0) |

The `Owner` column is already numeric (0, 1, 2, 3) and used as-is.

### Step 6 — Train / Test Split
The cleaned dataset is split into **80% training** and **20% testing** using `train_test_split` with a fixed `random_state` for reproducibility. Features (`X`) are all columns except `Selling_Price`; the target (`y`) is `Selling_Price`.

### Step 7–9 — Model Training
Three regression models are trained and evaluated. See [Models Used](#models-used) below.

### Step 10 — Model Comparison
All three models are compared side-by-side using MAE, RMSE, and R² score. Bar charts visualise the comparison.

### Step 11 — Predict Your Own Car Price
An interactive section at the end of the notebook lets you enter your own car's details and receive a predicted price from each model.

---

## Models Used

### 1. Linear Regression (Baseline)
A simple baseline model that fits a straight line through the feature space. Fast to train but assumes a linear relationship between features and price — which the scatter plots show is not strictly the case.

- **Pros:** Fast, interpretable, good baseline
- **Cons:** Cannot capture non-linear relationships (e.g. price drops sharply after high mileage, not linearly)

### 2. Random Forest Regressor
An ensemble of decision trees that each vote on the predicted price. The final prediction is the average of all trees. Handles non-linear relationships well and is robust to outliers.

- **Key hyperparameters:** `n_estimators`, `max_depth`
- **Pros:** Handles non-linearity, resistant to overfitting, produces feature importances
- **Cons:** Slower than linear regression, less interpretable

### 3. XGBoost Regressor
A gradient-boosted tree model where each new tree corrects the errors of the previous one. Generally the most powerful of the three for tabular data.

- **Key hyperparameters:** `n_estimators`, `learning_rate`, `max_depth`
- **Pros:** High accuracy, fast training with GPU support, handles missing values natively
- **Cons:** More hyperparameters to tune, can overfit on small datasets

---

## Results

All three models are evaluated on the held-out test set using:

| Metric | Description |
|---|---|
| **MAE** (Mean Absolute Error) | Average absolute difference between predicted and actual price (in ₹ lakhs) |
| **RMSE** (Root Mean Squared Error) | Penalises large errors more heavily than MAE |
| **R² Score** | Proportion of price variance explained by the model (1.0 = perfect) |

> **Note:** Exact metric values appear in the notebook output after running all cells. Tree-based models (Random Forest and XGBoost) are expected to outperform Linear Regression significantly given the non-linear nature of the data.

---

## How to Run

### Option A — Google Colab (recommended, no setup needed)

1. Upload `rgressions.ipynb` to [colab.research.google.com](https://colab.research.google.com)
2. When prompted, also upload `car_data.csv` to the Colab session
3. Go to **Runtime → Run all**

### Option B — Local (Jupyter Notebook or VS Code)

**Step 1 — Clone or download the project**
```bash
git clone https://github.com/justsomeguy4411/Car-Price-Prediction.git
cd Car-Price-Prediction
```

**Step 2 — Install dependencies**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

**Step 3 — Launch the notebook**
```bash
jupyter notebook regressions.ipynb
```

**Step 4 — Run all cells**

Use **Kernel → Restart & Run All** to execute the full pipeline from top to bottom.

> Make sure `car_data.csv` is in the **same folder** as the notebook, or update the `FILE_PATH` variable in Step 2 of the notebook to point to the correct location.

---

## Requirements

| Package | Purpose |
|---|---|
| `pandas` | Data loading and manipulation |
| `numpy` | Numerical operations |
| `matplotlib` | Plotting |
| `seaborn` | Statistical visualisation |
| `scikit-learn` | Linear Regression, Random Forest, preprocessing, metrics |
| `xgboost` | XGBoost Regressor |
| `jupyter` | Running the `.ipynb` notebook |

Install all at once:
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

Python version: **3.8 or higher** recommended.

---

## Releases

### v1.0.0 — Initial Release
> 📅 April 2026

First stable version of the Car Price Prediction project, submitted as a term-ending group project.

**What's included in this release:**
- `rgressions.ipynb` — complete ML pipeline notebook (11 steps, fully documented)
- `car_data.csv` — cleaned CarDekho dataset (301 rows × 9 columns)
- `README.md` — full project documentation

**Models shipped in v1.0.0:**
- ✅ Linear Regression (baseline)
- ✅ Random Forest Regressor
- ✅ XGBoost Regressor

**Features:**
- End-to-end pipeline from raw CSV to price prediction
- EDA with 9 visualisation charts (histograms, scatter plots, bar charts, heatmap)
- Feature engineering (`Car_Age` derived from `Year`)
- Label encoding for all categorical columns
- Side-by-side model comparison (MAE, RMSE, R²)
- Interactive prediction cell — enter your own car specs and get an instant price estimate
- Compatible with Google Colab and local Jupyter

**How to download this release:**

Go to the [Releases](../../releases) tab on GitHub → click **v1.0.0** → download the `.zip` or `.tar.gz` source archive.

---

### Planned for v1.1.0
- [ ] `requirements.txt` file for one-command install
- [ ] Save trained models to disk using `joblib` (`.pkl` files)
- [ ] Add a simple web interface using Streamlit or Gradio
- [ ] Support for additional datasets (more car brands, newer years)
- [ ] Hyperparameter tuning with `GridSearchCV` for Random Forest and XGBoost

---

## Packages

This project does not publish a pip-installable Python package. The notebook and dataset are the deliverables and are used directly.

### Python Dependencies

All packages used in this project are listed below with their minimum recommended versions:

| Package | Version | Install command |
|---|---|---|
| `pandas` | ≥ 1.3.0 | `pip install pandas` |
| `numpy` | ≥ 1.21.0 | `pip install numpy` |
| `matplotlib` | ≥ 3.4.0 | `pip install matplotlib` |
| `seaborn` | ≥ 0.11.0 | `pip install seaborn` |
| `scikit-learn` | ≥ 0.24.0 | `pip install scikit-learn` |
| `xgboost` | ≥ 1.4.0 | `pip install xgboost` |
| `jupyter` | ≥ 1.0.0 | `pip install jupyter` |

**Install everything at once:**
```bash
pip install pandas numpy matplotlib seaborn scikit-learn xgboost jupyter
```

### Creating a `requirements.txt`

If you want to freeze your exact environment versions for reproducibility, run this inside your project folder:

```bash
pip freeze > requirements.txt
```

Then anyone can replicate your exact environment with:
```bash
pip install -r requirements.txt
```

### Conda Environment (alternative)

If you use Anaconda or Miniconda, create an isolated environment:

```bash
conda create -n carprice python=3.10
conda activate carprice
conda install pandas numpy matplotlib seaborn scikit-learn jupyter
pip install xgboost
jupyter notebook rgressions.ipynb
```

### Google Colab

No installation needed. All packages except `xgboost` are pre-installed in Colab. If `xgboost` is missing, add this cell at the top of the notebook:

```python
!pip install xgboost
```

---

## Team

Built by a student group as a term-ending project.

- **Contributors:** see GitHub contributors panel
- **Dataset credit:** [Zafarali27 — Kaggle](https://www.kaggle.com/datasets/zafarali27/car-price-prediction/data)
