# San Francisco Crime Classification

Multiclass classification of San Francisco crime incidents into 39 crime categories using spatiotemporal features. Built on the SFPD incident reports dataset (2003–2015) with ~878K records.

## Problem Statement

Given the time, location, and police district of a crime incident in San Francisco, predict the category of crime that occurred. This is a 39-class classification problem evaluated using multiclass log loss.

## Approach

### Feature Engineering
- **Spatial features**: Latitude/longitude coordinates, KMeans-based spatial clusters
- **Temporal features**: Hour-of-day bins, day-of-week, seasonal indicators, weekend flags
- **District features**: Police district encoding
- **Validation strategy**: Odd-week / even-week split to minimize temporal leakage

### Models Compared

| Model | Accuracy | Log Loss |
|-------|----------|----------|
| **XGBoost** | **23.09%** | **2.6315** |
| Logistic Regression | 22.41% | 2.5916 |
| Naive Bayes | 20.44% | 2.7133 |
| KNN | 15.15% | 17.2831 |
| Random Forest | 14.85% | 3.5197 |

### Model Interpretability
- SHAP values for global and local feature importance
- Partial Dependence Plots (PDP) for feature-target relationships
- Geographic heatmaps of crime density across SF

## Tech Stack

- **ML**: scikit-learn, XGBoost, LightGBM
- **Data**: pandas, NumPy
- **Visualization**: matplotlib, seaborn, geopandas, contextily, geoplot
- **Interpretability**: SHAP, ELI5, PDPbox

## Project Structure

```
├── 01_exploratory_data_analysis.ipynb   # Initial EDA & data profiling
├── 02_feature_engineering.ipynb         # Feature engineering iterations
├── 03_model_comparison.ipynb            # Model training & evaluation
├── 04_full_analysis_pipeline.ipynb      # End-to-end pipeline
└── README.md
```

## Dataset

The dataset is from the [SF Crime Classification](https://www.kaggle.com/c/sf-crime) Kaggle competition. Download `train.csv` and `test.csv` into the project root to reproduce results.

## Getting Started

```bash
pip install pandas numpy scikit-learn xgboost lightgbm matplotlib seaborn geopandas shap eli5 pdpbox
jupyter notebook 01_exploratory_data_analysis.ipynb
```
