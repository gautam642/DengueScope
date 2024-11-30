🌍 DengueScope: A Machine Learning-Based Dengue Prediction System




DengueScope is a machine learning-based system designed to predict dengue outbreaks using a diverse set of geospatial, climatic, socio-economic, and human mobility data. The project integrates novel data sources such as Google Trends and Twitter geotagged data to enhance dengue prediction accuracy across Brazilian cities.

🚀 Project Overview
Dengue is a major public health concern, especially in tropical regions like Brazil. DengueScope aims to predict dengue outbreaks by leveraging:

Geospatial Data: Population density, urbanization, and spatial spread of dengue cases.
Climatic Data: Temperature, humidity, precipitation, and other environmental factors.
Socio-economic Data: Income levels, healthcare access, and population vulnerability.
Human Mobility Data: Twitter geotagged data and Google Trends for human movement patterns and public awareness.
🛠️ Key Features
Dimensionality Reduction using Principal Component Analysis (PCA) for efficient data handling.
Advanced Machine Learning Models:
CatBoost (Gradient Boosting)
Long Short-Term Memory (LSTM) Networks
Temporal Convolutional Networks (TCN)
Temporal Fusion Transformer (TFT)
Ensemble Learning to combine model predictions for improved accuracy and generalization.
Future Exploration: Graph Neural Networks (GNNs) for spatial mobility analysis.
📊 Project Structure
bash
Copy code
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
📚 Data Sources
Geospatial and Climate Data:

Brazilian Institute of Geography and Statistics (IBGE)
World Meteorological Organization (WMO)
Socio-economic Data:

World Bank Open Data
Human Mobility Data:

Twitter geotagged data collected through Twitter API
Google Trends search data for dengue-related terms
🔍 Methodology
1. Data Integration
Merging geospatial, climate, socio-economic, and human mobility data.
Preprocessing and feature extraction using GROUPED_VARS and DATA_REDUCER_SETTINGS.
2. Model Development
CatBoost: Handles categorical data and complex feature interactions.
LSTM: Captures temporal dependencies in dengue outbreaks.
TCN: Focuses on time-series data with convolutional layers.
TFT: Integrates time-varying features and attention mechanisms.
3. Evaluation
Compare model performance using metrics like RMSE, MAE, F1-score, and AUC-ROC.
Analyze the impact of Google Trends and Twitter data on prediction accuracy.
4. Ensemble Learning
Combine predictions from CatBoost, LSTM, TCN, and TFT for better generalization.
5. Future Work
Explore Graph Neural Networks (GNNs) for spatial analysis.
Enhance human mobility modeling using additional social media and mobility datasets.
📈 Model Comparison
Model	RMSE	MAE	F1-Score	AUC-ROC
CatBoost	X.XX	X.XX	X.XX	X.XX
LSTM	X.XX	X.XX	X.XX	X.XX
TCN	X.XX	X.XX	X.XX	X.XX
TFT	X.XX	X.XX	X.XX	X.XX
Ensemble	X.XX	X.XX	X.XX	X.XX
📌 How to Use
Clone the repository:
bash
Copy code
git clone https://github.com/GAUTAMMANU/proj.git
cd proj
Install dependencies:
bash
Copy code
pip install -r requirements.txt
Run data preprocessing:
bash
Copy code
python src/data_loader.py
Train the models:
bash
Copy code
python src/model_train.py --model <catboost|lstm|tcn|tft>
Run ensemble learning:
bash
Copy code
python src/ensemble.py
📄 License
This project is licensed under the MIT License. See the LICENSE file for more details.

🤝 Contributing
Contributions are welcome! Feel free to fork the repository and submit a pull request or open an issue for any suggestions.

📫 Contact
For any questions or inquiries, feel free to reach out:

GitHub Issues
