# Notebooks Overview

This directory contains Jupyter notebooks focused on analyzing different aspects of data in a fraud data context. Below is a brief description of each notebooks.

## Notebooks

### 1. `exploratory_data_analysis.ipynb`

In this notebook I have included all my efforts to investigate the nature of the provided fraud data for Adey Innovations' users. I contains **Basic Summary Statistics**, **Univariate Analysis**, **Bivariate Analysis**, **Missing Value Analysis**, **Outlier Detection** and **Simple Fraud Analysis**

### 2. `feature_engineering.ipynb`

In this notebook I have included all my efforts of feature engineering. It contains the **Breakdown of Date feature**, **Handeling of Missing Values**, **Aggeregation of Data Per-Customer**, **Normalizing Numerical Data** and finally **Encoding of Categorical Features**.

### 3. `modeling.ipynb`

In this notebook I have included my efforts to train different classification model and then choose the best amongst them. I used **optuna** to choose the best parameters for the models, and used mlflow to log models and select the best one for the logged ones.

### 4. `model_explanation.ipynb`

In this notebook I have included my efforts to provide model explanation using SHAP and LIME modules.
