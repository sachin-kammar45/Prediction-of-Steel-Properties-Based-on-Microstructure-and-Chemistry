# Prediction of Steel Properties Based on Microstructure and Chemistry

A machine learning framework for predicting mechanical properties of steels (yield strength, tensile strength, and elongation) using chemical composition and microstructural features.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)

## 📋 Overview

This project leverages machine learning models to explore the relationships between alloying elements, microstructure, and mechanical performance in steels. By analyzing chemical composition and microstructural features, the models can predict key mechanical properties critical for materials design and engineering applications.

## ✨ Features

- **Multiple ML Models**: Gradient Boosting, Random Forest, Neural Networks, and ensemble methods for multi-output regression
- **Feature Importance Analysis**: Identify key chemical elements and microstructural features affecting mechanical properties
- **SHAP Analysis**: Interpretable AI to understand feature contributions to predictions
- **Comprehensive Visualizations**: 
  - Histograms and KDE plots for property distributions
  - Bar plots for feature importance
  - Line plots for property relationships
  - Regression plots for model performance
- **Trade-off Analysis**: Explore relationships between yield strength, tensile strength, and elongation
- **Simulated Chemical Composition**: Handle missing chemical data through intelligent simulation

## 🚀 Installation

### Clone the Repository

```bash
git clone https://github.com/yourusername/Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry.git
cd Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Required Packages

- `numpy`
- `pandas`
- `scikit-learn`
- `matplotlib`
- `seaborn`
- `xgboost`
- `shap`

## 📊 Usage

### 1. Load Data

```python
import pandas as pd

# Load the steel properties dataset
df = pd.read_csv("steel_strength.csv")
```

### 2. Preprocess and Split Data

```python
from sklearn.model_selection import train_test_split

# Define features and target columns
features = ['C', 'Mn', 'Cr', 'Ni', 'Mo', 'microstructure_features']
target_cols = ['Yield_Strength', 'Tensile_Strength', 'Elongation']

# Split the data
X_train, X_valid, y_train, y_valid = train_test_split(
    df[features], 
    df[target_cols], 
    test_size=0.2, 
    random_state=42
)
```

### 3. Train a Model

```python
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor

# Initialize and train the model
model = MultiOutputRegressor(GradientBoostingRegressor())
model.fit(X_train, y_train)

# Make predictions
predictions = model.predict(X_valid)
```

### 4. Visualize Results

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Feature importance plots
# Histograms and KDE for mechanical properties
# SHAP plots for interpretability
# Regression plots for model performance
```

## 📈 Results & Insights

### Key Findings

- **Top Influencing Elements**: Carbon (C), Manganese (Mn), and Chromium (Cr) show the highest impact on mechanical properties
- **Property Relationships**: 
  - Yield strength positively correlates with tensile strength
  - Both strength metrics inversely correlate with elongation
- **Design Implications**: The analysis highlights which chemical elements and microstructural features are most impactful for alloy design

### Model Performance

The models achieve strong predictive performance across all three mechanical properties, with detailed metrics available in the analysis notebooks.



## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request. For major changes, please open an issue first to discuss what you would like to change.

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🔬 Applications

This work has applications in:
- Steel alloy design and optimization
- Materials science research
- Manufacturing process optimization
- Quality control and prediction
- Academic research in metallurgy

## 📚 Citation

If you use this code in your research, please cite:

```bibtex
@software{steel_properties_prediction,
  author = {Your Name},
  title = {Prediction of Steel Properties Based on Microstructure and Chemistry},
  year = {2024},
  url = {https://github.com/yourusername/Prediction-of-Steel-Properties-Based-on-Microstructure-and-Chemistry}
}
```

## 📧 Contact

For questions or collaborations, please open an issue or contact [your-email@example.com]

## 🏷️ Keywords

`Steel` `Mechanical Properties` `Yield Strength` `Tensile Strength` `Elongation` `Chemical Composition` `Microstructure` `Machine Learning` `SHAP` `Feature Importance` `Alloy Design` `Materials Science` `Predictive Modeling`

---

⭐ If you find this project useful, please consider giving it a star!
