# 🏢 Building Energy Efficiency Predictor
> **ADS FA2 Case Study** — Data Analysis Using Machine Learning & Streamlit

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://building-energy-efficiency-predictor.streamlit.app)
![Python](https://img.shields.io/badge/Python-3.x-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.8-orange?logo=scikit-learn)
![Streamlit](https://img.shields.io/badge/Streamlit-1.64-red?logo=streamlit)

---

## 📌 Project Overview

This project applies **4 different Machine Learning algorithms** to the **UCI Energy Efficiency Dataset** to predict and analyze the energy requirements of residential buildings.

Given 8 building parameters (compactness, surface area, wall area, roof area, height, orientation, glazing area, glazing distribution), the models predict:

| # | Algorithm | Task | Type | Accuracy |
|---|---|---|---|---|
| 1 | **Linear Regression** | Predict exact Heating Load (kWh/m²) | Regression | R² = 91.22% |
| 2 | **Random Forest** | Classify Heating Load as High or Low | Binary Classification | 99.35% |
| 3 | **K-Nearest Neighbors (KNN)** | Predict Cooling Load class (Low/Medium/High) | Multi-class Classification | 92.21% |
| 4 | **K-Means Clustering** | Group buildings by energy efficiency profile | Unsupervised Clustering | K=4 clusters |

---

## 📊 Dataset

**Source:** [UCI Machine Learning Repository — Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)

| Property | Details |
|---|---|
| Records | 768 buildings |
| Features | 8 input features |
| Targets | Y1 = Heating Load, Y2 = Cooling Load |
| Missing Values | None |

### Features

| Feature | Description | Range |
|---|---|---|
| Relative Compactness | Ratio of building volume to surface area | 0.62 – 0.98 |
| Surface Area (m²) | Total building surface area | 514 – 808 |
| Wall Area (m²) | Total wall area | 245 – 416 |
| Roof Area (m²) | Total roof area | 110 – 220 |
| Overall Height (m) | Building height | 3.5 or 7.0 |
| Orientation | Building direction (N/E/S/W) | 2, 3, 4, 5 |
| Glazing Area | Window area as % of floor | 0%, 10%, 25%, 40% |
| Glazing Distribution | How windows are distributed | 0–5 |

---

## 🤖 ML Models & Results

### 1. Linear Regression — Predict Heating Load
- **Task:** Predict the exact heating energy requirement (kWh/m²)
- **Train R²:** 91.71% | **Test R²:** 91.22%
- **MAE:** 2.18 kWh/m² | **RMSE:** 3.03 kWh/m²

### 2. Random Forest Classifier — High/Low Heating Load
- **Task:** Classify if a building has High or Low heating requirement
- **Threshold:** Median Heating Load = 18.95 kWh/m²
- **Train Accuracy:** 100% | **Test Accuracy:** 99.35%
- **Precision:** 1.00 | **Recall:** 0.988 | **F1:** 0.994
- **Tuning:** GridSearchCV (n_estimators, max_depth)

### 3. KNN Classifier — Cooling Load Category
- **Task:** Predict cooling load class (Low / Medium / High)
- **Best K:** 1 | **Train Accuracy:** 100% | **Test Accuracy:** 92.21%
- **Weighted F1:** 0.9225

### 4. K-Means Clustering — Building Energy Profiles
- **Task:** Group buildings into energy efficiency clusters
- **Optimal K:** 4 (found using Elbow Method)
- **Clusters:** High Efficiency | Medium Efficiency | Low Efficiency | Very Low Efficiency

---

## 🖥️ Streamlit Web Application

The interactive web app allows users to enter building specifications and instantly get predictions from all 4 ML models.

### Features
- 📈 **Heating Load Prediction** (Linear Regression)
- 🌲 **High/Low Heating Classification** with probability (Random Forest)
- 📍 **Cooling Load Category** with bar chart (KNN)
- 🔵 **Building Energy Cluster** (K-Means)
- 📋 **Summary Table** of all predictions

### Run Locally
```bash
# Clone the repo
git clone https://github.com/SwapnilNarwade1203/Building-Energy-Efficiency-Predictor.git
cd Building-Energy-Efficiency-Predictor

# Install dependencies
pip install -r requirements.txt

# Run the notebook first to train models
jupyter notebook 125M1H045_ADS_FA2.ipynb

# Launch Streamlit app
streamlit run app.py
```

---

## 📁 Project Structure

```
Building-Energy-Efficiency-Predictor/
│
├── 125M1H045_ADS_FA2.ipynb     # Jupyter Notebook (EDA + Model Training)
├── app.py                       # Streamlit Web Application
├── requirements.txt             # Python dependencies
├── README.md                    # Project documentation
│
├── dataset/
│   └── energy_efficiency.xlsx   # UCI Energy Efficiency Dataset
│
├── models/
│   ├── linear_regression.pkl    # Trained Linear Regression model
│   ├── random_forest.pkl        # Trained Random Forest model
│   ├── knn_model.pkl            # Trained KNN model
│   ├── kmeans_model.pkl         # Trained K-Means model
│   ├── median_hl.pkl            # Heating load threshold
│   └── knn_thresholds.pkl       # Cooling load thresholds
│
└── scaler/
    └── scaler.pkl               # StandardScaler for feature normalization
```

---

## ⚙️ Installation & Requirements

```bash
pip install -r requirements.txt
```

**Dependencies:**
- `pandas` — Data manipulation
- `numpy` — Numerical operations
- `matplotlib` — Plotting
- `seaborn` — Statistical visualization
- `scikit-learn` — Machine learning models
- `streamlit` — Web application
- `joblib` — Model serialization
- `openpyxl` — Excel file reading

---

## 📷 App Screenshots

> Run the app locally at `http://localhost:8501` after setup.

---

## 📚 References

- [UCI Energy Efficiency Dataset](https://archive.ics.uci.edu/ml/datasets/Energy+efficiency)
- [scikit-learn Documentation](https://scikit-learn.org/)
- [Streamlit Documentation](https://docs.streamlit.io/)
- Tsanas, A. and Xifara, A. (2012) *Accurate quantitative estimation of energy performance of residential buildings using statistical machine learning tools*, Energy and Buildings.
