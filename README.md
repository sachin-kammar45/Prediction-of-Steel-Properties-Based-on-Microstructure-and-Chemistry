# Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry
This repository contains code and analysis for predicting mechanical properties of steels (yield strength, tensile strength, and elongation) based on chemical composition and microstructural features.
Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry

This repository contains code and analysis for predicting mechanical properties of steels — including yield strength, tensile strength, and elongation — based on chemical composition and microstructural features. The project leverages machine learning models to explore the relationships between alloying elements, microstructure, and mechanical performance.

Features

Machine Learning Models: Gradient Boosting, Random Forest, Neural Networks, and more for multi-output regression.

Feature Importance Analysis: Identify key chemical elements and microstructural features affecting mechanical properties.

SHAP Analysis: Understand contributions of each feature to predictions.

Visualizations: Histograms, KDEs, bar plots, line plots, and regression plots for thesis-ready figures.

Trade-off Analysis: Explore relationships between yield strength, tensile strength, and elongation.

Simulated Chemical Composition: For cases where chemical data is missing, simulated alloying elements are used for analysis.

Installation
git clone https://github.com/yourusername/Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry.git
cd Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry
pip install -r requirements.txt


Recommended packages: numpy, pandas, scikit-learn, matplotlib, seaborn, xgboost, shap

Usage

Load Data:

import pandas as pd

df = pd.read_csv("steel_strength.csv")


Preprocess and Split Data:

X_train, X_valid, y_train, y_valid = train_test_split(df[features], df[target_cols], test_size=0.2, random_state=42)


Train a Model:

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(X_train, y_train)


Visualize Feature Importance and Relationships:

Histograms, KDEs, and bar charts for mechanical properties

Line plots for property relationships

SHAP plots for chemical feature importance

Results & Insights

Top Influencing Elements: Carbon (C), Manganese (Mn), Chromium (Cr)

Property Relationships: Yield strength positively correlates with tensile strength; both inversely correlate with elongation.

Design Implication: Highlights which chemical elements and microstructural features are most impactful for alloy design.

License

MIT License – see LICENSE

Keywords

Steel, Mechanical Properties, Yield Strength, Tensile Strength, Elongation, Chemical Composition, Microstructure, Machine Learning, SHAP, Feature Importance, Alloy Design
