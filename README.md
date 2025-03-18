# 🌍 **DengueScope: A Machine Learning-Based Dengue Prediction System**  

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)  

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

## 📚 **Data Sources**

The data used in this project comes from a variety of reliable sources:

1. **Geospatial and Climate Data**:  
   - [Brazilian Institute of Geography and Statistics (IBGE)](https://www.ibge.gov.br)  
   - [World Meteorological Organization (WMO)](https://public.wmo.int)  

2. **Socio-economic Data**:  
   - [World Bank Open Data](https://data.worldbank.org)

3. **Human Mobility Data**:  
   - Twitter geotagged data collected using the [Twitter API](https://developer.twitter.com/en/docs/twitter-api)  
   - Google Trends search data for dengue-related terms, processed using serpapi

4. **https://github.com/ESA-PhiLab/ESA-UNICEF_DengueForecastProject** -preprocessing and inspiration.
---

## 🔍 **Methodology**

The DengueScope project follows a systematic approach to data integration, model development, and evaluation:

### 1. **Data Integration**  
   - Merge geospatial, climate, socio-economic, and human mobility data to create a unified dataset.

### 2. **Model Development**  
   Four machine learning models are implemented to predict dengue outbreaks:  

   - **CatBoost**: A gradient boosting algorithm that efficiently handles categorical data and complex feature interactions.  
   - **LSTM (Long Short-Term Memory)**: A recurrent neural network (RNN) variant designed to capture temporal dependencies in time-series data.  
   - **TCN (Temporal Convolutional Network)**: A deep learning model that leverages convolutional layers for sequential data analysis.  
   - **TFT (Temporal Fusion Transformer)**: An advanced model that combines time-varying features and attention mechanisms for multivariate time-series forecasting.

### 3. **Evaluation**  
   - Evaluate each model's performance using key metrics such as:  
     - **Root Mean Square Error (RMSE)**  
     - **Mean Absolute Error (MAE)**  
     - **F1-Score**  
   - Analyze the impact of integrating Google Trends and Twitter data on model accuracy and generalization.

### 4. **Ensemble Learning**  
   - Combine the predictions from all four models using an ensemble approach to improve overall accuracy and robustness.

### 5. **Future Exploration**  
   - Explore the use of **Graph Neural Networks (GNNs)** to analyze spatial relationships in human mobility and dengue spread patterns.  
   - Incorporate additional mobility data sources for enhanced predictive capabilities.

---
## 📌 **How to Use**

### a. Clone the repository on you desktop

This operation can be done in two ways:

1. open yout terminal, navigate to you desktop and run 
   `git https://github.com/ESA-PhiLab/ESA-UNICEF_DengueForecastProject`
2. download the project on your desktop

### b. Create a new virtual environment
This allows to create a isolated environment and to control the installation of dependancies without affecting your python base installation.
Before creating a new environment, be sure that the default anaconda enviroment is not active. By opening a terminal, you just need to check if before you user name there is something like ` (base) username: $`. If so run this command first:

` conda deactivate base ` or ` conda deactivate <environmentname> `

otherwise run directly this command, that creates a new enviroment (change yourenvname with whaterver you prefer)

` conda create -n yourenvname python=x.x anaconda` (3.9 or later)

### c. Acticate the virtual environment

To work on the environment you need to activate it first

` conda activate yourenvname `

### d. Install dependancies

First thing first, you need to move to the project folder

` cd <path to the project folder> `, for example ` cd Desktop\ESA-UNICEF_DengueForecastProject`

then you need to install pip (python package manager) on the conda environment

` conda install pip `

Then you can install the remaining packages listed in *[requirements.txt](../code/requirements.txt)*

` pip install code\requirements.txt `

main files are - 
https://github.com/GAUTAMMANU/proj/blob/main/code/modelling-Brazil-new2.ipynb (no-search trends data integration)
https://github.com/GAUTAMMANU/proj/blob/main/code/modelling-Brazil-new3.ipynb (trends data integrated and used)
