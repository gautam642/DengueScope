# 🌍 **DengueScope: A Machine Learning-Based Dengue Prediction System**  

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)  
[![CatBoost](https://img.shields.io/badge/CatBoost-1.0-green)](https://catboost.ai/)  
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)  
[![License](https://img.shields.io/github/license/GAUTAMMANU/proj)](LICENSE)  

DengueScope is a machine learning-based system designed to predict dengue outbreaks using a diverse set of geospatial, climatic, socio-economic, and human mobility data. The project integrates novel data sources such as Google Trends and Twitter geotagged data to enhance dengue prediction accuracy across Brazilian cities.

## 🚀 **Project Overview**

Dengue is a major public health concern, especially in tropical regions like Brazil. DengueScope aims to predict dengue outbreaks by leveraging:

- **Geospatial Data**: Population density, urbanization, and spatial spread of dengue cases.
- **Climatic Data**: Temperature, humidity, precipitation, and other environmental factors.
- **Socio-economic Data**: Income levels, healthcare access, and population vulnerability.
- **Human Mobility Data**: Twitter geotagged data and Google Trends for human movement patterns and public awareness.

## 🛠️ **Key Features**

- **Dimensionality Reduction** using Principal Component Analysis (PCA) for efficient data handling.
- **Advanced Machine Learning Models**:  
  - CatBoost (Gradient Boosting)  
  - Long Short-Term Memory (LSTM) Networks  
  - Temporal Convolutional Networks (TCN)  
  - Temporal Fusion Transformer (TFT)  
- **Ensemble Learning** to combine model predictions for improved accuracy and generalization.
- **Future Exploration**: Graph Neural Networks (GNNs) for spatial mobility analysis.

## 📊 **Project Structure**  

```bash
├── data/                   # Dataset files (geospatial, climate, socio-economic, etc.)
├── models/                 # Machine learning models (CatBoost, LSTM, TCN, TFT)
├── notebooks/              # Jupyter notebooks for data preprocessing and experimentation
├── src/                    # Source code for data processing and model implementation
│   ├── data_loader.py      # Load and preprocess datasets
│   ├── feature_engineering.py  # Feature extraction and dimensionality reduction
│   ├── model_train.py      # Training scripts for individual models
│   ├── ensemble.py         # Ensemble learning implementation
├── results/                # Model results and performance metrics
├── README.md               # Project documentation
└── LICENSE                 # License file

