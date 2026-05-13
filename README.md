# Medical Disease Classification

## Overview
Built a multi-class disease classification system using symptom data.

## Dataset
Diseases and Symptoms Dataset:
https://www.kaggle.com/datasets/dhivyeshrk/diseases-and-symptoms-dataset

## Models Tested
- Neural Network (PyTorch)
- Random Forest
- XGBoost

## Results
Random Forest Classifier Performance:

- Accuracy: 84.0%
- Macro F1 Score: 0.85
- Weighted F1 Score: 0.84
- Precision: 0.84
- Recall: 0.84

Dataset:
- 700+ disease classes
- symptom-based multiclass classification
- rare class filtering applied

## Preprocessing
- Missing values filled with 0
- Rare disease classes filtered
- Stratified train-test split

## Installation
pip install -r requirements.txt

## Future Improvements
- Hyperparameter tuning
- Macro F1 optimization
- Streamlit dashboard
