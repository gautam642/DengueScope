# 🌍 **DengueScope: A Multi-Modal Machine Learning Framework for Dengue Forecasting**  

[![Python](https://img.shields.io/badge/python-3.x-blue.svg)](https://www.python.org/)  

DengueScope is a machine learning-based system designed to predict dengue outbreaks using a diverse set of geospatial, climatic, socio-economic, and human mobility data. The project integrates novel data sources such as Google Trends alongside models such as TCN, TFT, CatBoost, LSTM, BlendeStackEnsemble  to enhance dengue prediction accuracy across Brazilian cities.

## 🚀 **Project Overview**

Dengue is a major public health concern, especially in tropical regions like Brazil. DengueScope aims to predict dengue outbreaks with maximum accuracy by leveraging:

- **Geospatial Data**: Population density, urbanization, and spatial spread of dengue cases.
- **Climatic Data**: Temperature, humidity, precipitation, and other environmental factors.
- **Socio-economic Data**: Income levels, healthcare access, and population vulnerability.
- **Google Searches**: Google Trends data
- **Interactive Maps**: Visual representation of dengue spread across Brazilian states

## 📚 **Project Workflow**

### 1. **Data Preprocessing and Initial Modeling**
- Initial data processing and feature engineering in `modelling-brazil-new3.ipynb`
- Data augmentation and preprocessing steps
- Training and saving of base models
- Metrics calculation and storage

### 2. **Google Trends Integration**
- Data collection and processing using SerpAPI in `google_trends/serpapi_trends.ipynb`
- Filtering, cleaning, and scaling of Google Trends data
- Creation of merged dataset with original features
- Retraining of models with enhanced dataset

### 3. **Advanced Feature Engineering and Analysis**
- Comprehensive analysis in `google_trends/analysis.ipynb`
- Feature importance analysis
- Implementation of various techniques(used Granger causality, Cross-Correlation):
  - Lagged features
  - Exponentially Weighted Moving Averages (EWMA)
  - Rolling averages
- Dataset version comparison:
  - Used simpler models like to draw comparisons among various versions of the datset
    -models:
      - SARIMAX Model
      - Gradient Boosting Regressor
      - Random Forest Regressor
    - versions:
      - with added lag features, rollingavg+lag, ewma+lag
      - with added lag features only; NO rollingavg+lag, ewma+lag
      - with added lag features rollingavg+lag, ewma+lag and orignial features removed for which new versions were added
      - with added lag features only; NO rollingavg+lag, ewma+lag and orignial features removed for which new versions were added
      - original no lags
      - original no lags no google trends terms
      - original with only google terms lagged
      - with added lag features, rollingavg+lag, ewma+lag, mosquito_interest dropped (etc.)
- Final dataset selection: `merged_dataset_lagged.csv` (with added lag features, rollingavg+lag, ewma+lag)
- Details of simple models' metrics on various dataset versions in `baseline_metrics/baseline_model_metrics.txt`

### 4. **Model Training and Evaluation**
- Training of four main models on enhanced dataset and metrics calculation
- Comparison among each model's versions (trained of different datasets)
   - original
   - original + search terms
   - original + search terms + lagged features + ewma features + rollavg features
- Visualization and analysis in `metrics_visualiser.ipynb`
   - Overall model comparison (NRMSE, MAE) - boxplots
   - State-wise Model Comparison (NRMSE & MAE) - barplot
   - Facet Grid of Metrics Across Models and States
   - Taylor diagrams for model comparison 
      -all yielded best model - lagged dataset trained model except LSTM(original dataset model-best)
   - Interactive maps showing dengue spread and predictions & MAE
- Selection of best model versions

### 5. **Ensemble: BlendedStackingEnsemble**
- custom ensemble model combining both stacking and blending techniques
- Base Models: TCN, LSTM, CatBoost, and TFT
- Meta-Model: LightGBM trained on base model predictions (stacking)
- Blending: Weighted averaging of base model predictions
- Final Prediction: final_prediction = blend_ratio * meta_model_prediction + (1 - blend_ratio) * linear_combination
- Uses Optuna for hyperparameter(base_pred_weights, blend_ratio) tuning
- Initial implementation (Fixed weights and blend ratio) in `ensemble.py` , `train_ensemble.py`
- Advanced ensemble (optuna) in `ensemble_new.py` , `train_Ensemble_new.py`
- Final ensemble model comparison in `metrics_visualiser.ipynb`
- State-wise predictions saved in `raw_ensemble_predictions_Brazil.csv`

### 6. **Interactive Maps Visualization**
- **Geospatial Analysis**:
  - Choropleth maps showing dengue spread across Brazilian states
  - Time-series visualization of disease progression (Time slider for temporal analysis)
  - Comparison of actual vs predicted cases
  - Hover information for detailed statistics(State-Wise)
  - Built using Plotly


## 🛠️ **Key Features**

- **Multi-Modal Data Integration**:
  - Traditional epidemiological data (case counts, population density)
  - Climatic data (temperature, humidity, precipitation)
  - Digital signals (Google Trends, Twitter data)
  - Geospatial data (spatial distribution patterns)
  - Interactive maps for visual analysis

- **Advanced Machine Learning Models**:
  - **CatBoost**: Gradient boosting algorithm optimized for categorical features
  - **LSTM (Long Short-Term Memory)**: Deep learning model for time-series forecasting
  - **TCN (Temporal Convolutional Network)**: Convolution-based model for sequential data
  - **Ensemble Learning**: Blended stacking approach

- **Data Processing Pipeline**:
  - Automated data collection and preprocessing
  - Feature engineering and dimensionality reduction
  - Time-series analysis and feature extraction
  - Model training and validation

- **Evaluation Metrics**:
  - Root Mean Squared Error (RMSE)
  - Mean Absolute Error (MAE)
  - R-squared (R²)
  - Time-series specific metrics
  - Taylor diagrams for model comparison

## 📚 **Data Sources and Processing**

The system integrates data from multiple reliable sources:
- https://github.com/ESA-PhiLab/ESA-UNICEF_DengueForecastProject/blob/main/code/dataset/Brazil_UF_dengue_monthly.csv
- Google trends - using SerpAPI (https://trends.google.com/trends/),(https://serpapi.com/)


## 📌 **Getting Started**

### Prerequisites
- Python 3.9 or later
- Anaconda distribution recommended
- Internet connection for data collection

### Installation

1. **Clone the Repository**
```bash
git clone https://github.com/GAUTAMMANU/proj.git
cd proj
```

2. **Create Virtual Environment**
```bash
# Create conda environment
conda create -n denguescope python=3.9
conda activate denguescope

# Install dependencies
pip install -r requirements.txt
```

3. **Set Up API Keys**
Create a `.env` file in the root directory with your API keys:
```
SERPAPI_KEY1=your_key_1
SERPAPI_KEY2=your_key_2
SERPAPI_KEY3=your_key_3
```

### Usage

The main entry points are:
- [`code/train_ensemble_new.py`](code/train_ensemble_new.py): For training the ensemble model
- [`code/modelling-Brazil-new2.ipynb`](code/modelling-Brazil-new2.ipynb): Without Google Trends integration
- [`code/modelling-Brazil-new3.ipynb`](code/modelling-Brazil-new3.ipynb): With Google Trends integration
- [`google_trends/serpapi_trends.ipynb`](google%20trends/serpapi_trends.ipynb): Google Trends data processing
- [`google_trends/analysis.ipynb`](google%20trends/analysis.ipynb): Feature analysis and dataset preparation
- [`metrics_visualiser.ipynb`](metrics_visualiser.ipynb): Model performance visualization
- [`maps_visualiser.ipynb`](maps.ipynb): Interactive maps for dengue spread visualization ([``](dengue_brazil_with_metrics.html))

## 📈 **Results**

The ensemble approach shows significant improvements over individual models:
- Better handling of temporal patterns
- Improved prediction accuracy
- More robust to data variations
- Better generalization across different regions
- Enhanced performance with Google Trends integration
- Optimal feature combination through lagged features

## 🤝 **Contributing**

We welcome contributions to enhance the system. Please follow these guidelines:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Create a Pull Request


## 🙏 **Acknowledgments**

- Thanks to ESA-PhiLab for their preprocessing code inspiration
- Special thanks to the Brazilian health authorities for providing epidemiological data
- Gratitude to the open-source community for their contributions to machine learning libraries


## 🔮 **Future Scope**

### 1. **Graph Neural Networks (GNN) Integration**
- **Spatial-Temporal GNNs**:
  - Implement GraphSAGE or GAT (Graph Attention Networks) for capturing spatial dependencies between regions
  - Develop custom GNN layers for handling both spatial and temporal features
  - Integration with existing models for enhanced prediction accuracy

- **Multi-Scale Graph Learning**:
  - Hierarchical graph structures for different administrative levels
  - Cross-scale information propagation
  - Dynamic graph construction based on disease spread patterns

### 2. **Enhanced Mobility Data Integration**
- **Real-time Mobility Patterns**:
  - Integration of anonymized mobile phone data
  - GPS trajectory analysis
  - Public transportation flow data
  - Air travel network analysis

- **Advanced Mobility Features**:
  - Population movement matrices
  - Commuting patterns
  - Seasonal migration patterns
  - Event-driven mobility changes



## 📞 **Contact**

For any questions or issues, please open an issue in the GitHub repository.
