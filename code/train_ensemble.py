import os
import glob
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error
from datetime import datetime
import torch
import tensorflow as tf
from os.path import join

from ensemble import BlendedStackingEnsemble
from models import CatBoostNet, LSTMNet
from config import DATA_PROCESSING_SETTINGS
from datasetHandler import datasetHandler
# from config import config

# Import TCN and TFT model classes from the notebook
from tensorflow.keras import layers, models, regularizers, Input
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# Define whether to train a new model or load an existing one
# Set to True to train a new ensemble model
# Set to False to load a pre-trained model and make predictions
from os.path import join
import joblib



project_path = os.path.dirname(os.getcwd())
config = {
    'main_brazil': 'Brazil',
    'main_peru': 'Peru',
    'baseline': join(project_path, "baseline_models"),
    'output': join(project_path, "code", "saved_models"),
    'metrics': join(project_path, "code", "metrics")
}
[os.makedirs(val, exist_ok=True) for key, val in config.items()]

TRAINING = False

# Define TCN class from the notebook
class ImprovedTCNNet:
    def __init__(self, shape, output_units=2):
        self.shape = shape
        self.model = None
    
    def load(self, model_path):
        self.model = tf.keras.models.load_model(model_path) #only this line needed
        print(f"TCN model loaded successfully from {model_path}")

# Define TFT class from the notebook
class TemporalFusionTransformer(torch.nn.Module):
    def __init__(self, input_dim, output_dim=2, hidden_size=128, num_heads=8):
        super(TemporalFusionTransformer, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = torch.nn.LSTM(input_dim, hidden_size, batch_first=True, dropout=0.4, num_layers=1)
        self.attention = torch.nn.MultiheadAttention(hidden_size, num_heads, batch_first=True, dropout=0.2)
        self.layer_norm = torch.nn.LayerNorm(hidden_size)
        self.output_dense = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_size, output_dim)
        )

    def forward(self, x):
        # LSTM processing
        lstm_out, _ = self.lstm(x)
        # Attention layer
        attn_output, _ = self.attention(lstm_out, lstm_out, lstm_out)
        attn_output = self.layer_norm(attn_output + lstm_out)  # Residual connection
        # Final output layer
        output = self.output_dense(attn_output[:, -1, :])  # Use the last time step
        return output

def root_mean_squared_error(y_true, y_pred):
    return np.sqrt(mean_squared_error(y_true, y_pred))

# class DatasetHandler:
#     def prepare_data_LSTM(self, x_train, y_train, x_val, y_val):
#         return (x_train, y_train), (x_val, y_val)
    
#     def prepare_data_CatBoost(self, x_train, y_train, x_val, y_val):
#         x_train = x_train.reshape((x_train.shape[0], x_train.shape[1]*x_train.shape[2]))
#         x_val = x_val.reshape((x_val.shape[0], x_val.shape[1]*x_val.shape[2]))
#         return (x_train, y_train), (x_val, y_val)

def load_dataset_and_models():
    """Load dataset, training/validation data, and individual models"""
    print("Loading dataset...")
    
    # First, we need to get the dataset
    try:
        df_path = '../google trends/merged_dataset_lagged.csv'
        reduced_dataframe = pd.read_csv(df_path)
    except FileNotFoundError:
        # Try alternate path if the first one fails
        df_path = 'google trends/merged_dataset_lagged.csv'
        reduced_dataframe = pd.read_csv(df_path)
    except:
        print("Error loading dataset. Please check if file exists and path is correct.")
        raise
    
    # Split into training and validation
    training_dataframe = reduced_dataframe[reduced_dataframe.Year <= 2017]
    validation_dataframe = reduced_dataframe[reduced_dataframe.Year >= 2017]
    
    print(f"Training data: {len(training_dataframe)} samples")
    print(f"Validation data: {len(validation_dataframe)} samples")
    
    # Create dataset handler
    dataset_handler = datasetHandler(training_dataframe, validation_dataframe)
    
    # Prepare data
    # train_indices = training_dataframe.index.tolist()
    # val_indices = validation_dataframe.index.tolist()
    
    # Process target variables
    y_train = training_dataframe[['DengRate_all', 'DengRate_019']].values
    y_val = validation_dataframe[['DengRate_all', 'DengRate_019']].values
    
    # Get feature columns (excluding targets and specific columns)
    exclude_cols = ['DengRate_all', 'DengRate_019', 'Year', 'dep_id', 'Month']
    feature_cols = [col for col in reduced_dataframe.columns if col not in exclude_cols]
    
    # Process input features
    x_train = np.array(training_dataframe[feature_cols].values)
    x_val = np.array(validation_dataframe[feature_cols].values)
    
    x_train, y_train, x_val, y_val, train_indices, val_indices = dataset_handler.get_data(
        DATA_PROCESSING_SETTINGS['T LEARNING'],
        DATA_PROCESSING_SETTINGS['T PREDICTION']
    )
    
    return (dataset_handler, training_dataframe, validation_dataframe, 
            x_train, y_train, x_val, y_val, 
            train_indices, val_indices)

def main():
    print(f"Running in {'TRAINING' if TRAINING else 'INFERENCE'} mode")
    
    # Define paths
    output_path = os.path.join(config['output'], "Brazil")
    metrics_path = os.path.join(config['metrics'], "Brazil")
    os.makedirs(output_path, exist_ok=True)
    os.makedirs(metrics_path, exist_ok=True)

    # Load dataset and models
    try:
        (dataset_handler, training_dataframe, validation_dataframe, 
         x_train, y_train, x_val, y_val, 
         train_indices, val_indices) = load_dataset_and_models()
    except Exception as e:
        print(f"Error loading dataset: {e}")
        return
    
    # Create index mappings for department-level metrics
    y_val_indices_df = pd.DataFrame(val_indices, columns=['actual_index'])
    y_train_indices_df = pd.DataFrame(train_indices, columns=['actual_index'])
    
    # Define department names mapping
    DEP_NAMES = {11: 'Rondônia', 12: 'Acre', 13: 'Amazonas', 14: 'Roraima', 15: 'Pará', 
                16: 'Amapá', 17: 'Tocantins', 21: 'Maranhão', 22: 'Piauí', 23: 'Ceará',
                24: 'Rio Grande do Norte', 25: 'Paraíba', 26: 'Pernambuco', 27: 'Alagoas',
                28: 'Sergipe', 29: 'Bahia', 31: 'Minas Gerais', 32: 'Espírito Santo',
                33: 'Rio de Janeiro', 35: 'São Paulo', 41: 'Paraná', 42: 'Santa Catarina',
                43: 'Rio Grande do Sul', 50: 'Mato Grosso do Sul', 51: 'Mato Grosso',
                52: 'Goiás', 53: 'Distrito Federal'}
    
    if TRAINING:
        print("Training mode: Loading base models and generating predictions...")
        
        try:
            # Load base models - 
            #TCN - CORRECT
            print("Loading TCN model...")
            tcn_models = glob.glob(os.path.join(output_path, "TCN-new-lagged-*.keras"))
            if not tcn_models:
                print('No TCN model found. Cannot proceed with ensemble training.')
                return
            tcn = ImprovedTCNNet(shape=None)
            tcn.load(max(tcn_models))
            
            #LSTM - CORRECT
            print("Loading LSTM model...")
            lstm_models = glob.glob(os.path.join(output_path, "LSTM-new-lagged-*.h5"))
            if not lstm_models:
                print('No LSTM model found. Cannot proceed with ensemble training.')
                return
            # Get the shape from the training data
            trainT, valT = dataset_handler.prepare_data_LSTM(x_train, y_train, x_val, y_val)
            lstm = LSTMNet(shape=trainT[0].shape[1:])
            lstm.load(max(lstm_models))
            
            #CatBoost - CORRECT
            print("Loading CatBoost model...")
            catboost_models = glob.glob(os.path.join(output_path, "CATBOOST-lagged-*"))
            if not catboost_models:
                print('No CatBoost model found. Cannot proceed with ensemble training.')
                return
            catboost = CatBoostNet()
            catboost.load(max(catboost_models))
            
            #TFT - CORRECT
            print("Loading TFT model...")
            tft_models = glob.glob(os.path.join(output_path, "TFT_model_lagged_*.pt"))
            if not tft_models:
                print('No TFT model found. Cannot proceed with ensemble training.')
                return
            
            # Get input dimensions for TFT from sliced data
            input_dim = x_train[:, :, 2:].shape[2]  # Sliced feature size
            output_dim = y_train.shape[1]  # Target dimension
            
            # Load TFT model with correct dimensions and hyperparameters from notebook
            tft = TemporalFusionTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_size=64,  # Match notebook's hyperparameters
                num_heads=4      # Match notebook's hyperparameters
            )
            tft.load_state_dict(torch.load(max(tft_models), weights_only=True))
            tft.eval()
            
            # Process data for each model type
            print("Preparing data for predictions...")
            trainT, valT = dataset_handler.prepare_data_LSTM(x_train[:,:,2:], y_train, x_val[:,:,2:], y_val)
            trainC, valC = dataset_handler.prepare_data_CatBoost(x_train[:,:,2:], y_train, x_val[:,:,2:], y_val)
            
            # Generate predictions
            print("Generating predictions from individual models...")
            
            # TCN predictions
            tcn_preds_train = tcn.model.predict(trainT[0])
            tcn_preds_train[tcn_preds_train < 0] = 0
            tcn_preds_val = tcn.model.predict(valT[0])
            tcn_preds_val[tcn_preds_val < 0] = 0
            
            # LSTM predictions
            lstm_preds_train = lstm.model.predict(trainT[0])
            lstm_preds_train[lstm_preds_train < 0] = 0
            lstm_preds_val = lstm.model.predict(valT[0])
            lstm_preds_val[lstm_preds_val < 0] = 0
            
            # CatBoost predictions
            catboost_preds_train = catboost.model.predict(trainC[0])
            catboost_preds_train[catboost_preds_train < 0] = 0
            catboost_preds_val = catboost.model.predict(valC[0])
            catboost_preds_val[catboost_preds_val < 0] = 0
            
            # TFT predictions
            # Ensure the model dimensions match the data
            assert trainT[0].shape[-1] == input_dim, f"Mismatch: input_dim={input_dim}, trainT features={trainT[0].shape[-1]}"
            
            with torch.no_grad():
                tft_preds_train = tft(torch.tensor(trainT[0], dtype=torch.float32)).numpy()
                tft_preds_train[tft_preds_train < 0] = 0
                tft_preds_val = tft(torch.tensor(valT[0], dtype=torch.float32)).numpy()
                tft_preds_val[tft_preds_val < 0] = 0
            
            # Create base model predictions dictionary
            base_model_preds = {
                'tcn': tcn_preds_val,
                'lstm': lstm_preds_val,
                'catboost': catboost_preds_val,
                'tft': tft_preds_val
            }
            
            # Train ensemble model
            print("Training ensemble model...")
            ensemble = BlendedStackingEnsemble()
            ensemble.train_meta_model(base_model_preds, y_val)
            
            # Save the ensemble model
            ensemble.save_model(output_path)
            print("Ensemble model saved successfully")
            
            # Generate predictions using the trained ensemble
            ensemble_predictions = ensemble.predict(base_model_preds)
            
        except Exception as e:
            print(f"Error during training: {e}")
            import traceback
            traceback.print_exc()
            return
        
    else:
        print("Inference mode: Loading ensemble model and making predictions...")
        
        try:
            # Find the most recent ensemble model
            ensemble_models = glob.glob(os.path.join(output_path, "blended_stacking_*"))
            if not ensemble_models:
                print('No ensemble model found. Run with TRAINING = True first.')
                return
            
            # Load the most recent model
            latest_model = max(ensemble_models)
            print(f"Loading ensemble model from: {latest_model}")
            ensemble = BlendedStackingEnsemble()
            ensemble.load_model(latest_model)
            
            # Load base models
                        #TCN - CORRECT
            print("Loading TCN model...")
            tcn_models = glob.glob(os.path.join(output_path, "TCN-new-lagged-*.keras"))
            if not tcn_models:
                print('No TCN model found. Cannot proceed with ensemble training.')
                return
            tcn = ImprovedTCNNet(shape=None)
            tcn.load(max(tcn_models))
            
            #LSTM - CORRECT
            print("Loading LSTM model...")
            lstm_models = glob.glob(os.path.join(output_path, "LSTM-new-lagged-*.h5"))
            if not lstm_models:
                print('No LSTM model found. Cannot proceed with ensemble training.')
                return
            # Get the shape from the training data
            trainT, valT = dataset_handler.prepare_data_LSTM(x_train, y_train, x_val, y_val)
            lstm = LSTMNet(shape=trainT[0].shape[1:])
            lstm.load(max(lstm_models))
            
            #CatBoost - CORRECT
            print("Loading CatBoost model...")
            catboost_models = glob.glob(os.path.join(output_path, "CATBOOST-lagged-*"))
            if not catboost_models:
                print('No CatBoost model found. Cannot proceed with ensemble training.')
                return
            catboost = CatBoostNet()
            catboost.load(max(catboost_models))
            
            #TFT - CORRECT
            print("Loading TFT model...")
            tft_models = glob.glob(os.path.join(output_path, "TFT_model_lagged_*.pt"))
            if not tft_models:
                print('No TFT model found. Cannot proceed with ensemble training.')
                return
            
            # Get input dimensions for TFT from sliced data
            input_dim = x_train[:, :, 2:].shape[2]  # Sliced feature size
            output_dim = y_train.shape[1]  # Target dimension
            
            
            # Load TFT model with correct dimensions and hyperparameters from notebook
            tft = TemporalFusionTransformer(
                input_dim=input_dim,
                output_dim=output_dim,
                hidden_size=64,  # Match notebook's hyperparameters
                num_heads=4      # Match notebook's hyperparameters
            )
            tft.load_state_dict(torch.load(max(tft_models), weights_only=True))
            tft.eval()
            
            # Process data for each model type
            print("Preparing data for predictions...")
            trainT, valT = dataset_handler.prepare_data_LSTM(x_train[:,:,2:], y_train, x_val[:,:,2:], y_val)
            trainC, valC = dataset_handler.prepare_data_CatBoost(x_train[:,:,2:], y_train, x_val[:,:,2:], y_val)
            
            # Generate predictions
            print("Generating predictions from individual models...")
            
            # TCN predictions
            tcn_preds_train = tcn.model.predict(trainT[0])
            tcn_preds_train[tcn_preds_train < 0] = 0
            tcn_preds_val = tcn.model.predict(valT[0])
            tcn_preds_val[tcn_preds_val < 0] = 0
            
            # LSTM predictions
            lstm_preds_train = lstm.model.predict(trainT[0])
            lstm_preds_train[lstm_preds_train < 0] = 0
            lstm_preds_val = lstm.model.predict(valT[0])
            lstm_preds_val[lstm_preds_val < 0] = 0
            
            # CatBoost predictions
            catboost_preds_train = catboost.model.predict(trainC[0])
            catboost_preds_train[catboost_preds_train < 0] = 0
            catboost_preds_val = catboost.model.predict(valC[0])
            catboost_preds_val[catboost_preds_val < 0] = 0
            
            # Ensure the model dimensions match the data
            assert trainT[0].shape[-1] == input_dim, f"Mismatch: input_dim={input_dim}, trainT features={trainT[0].shape[-1]}"

            # TFT predictions
            with torch.no_grad():
                tft_preds_train = tft(torch.tensor(trainT[0], dtype=torch.float32)).numpy()
                tft_preds_train[tft_preds_train < 0] = 0
                tft_preds_val = tft(torch.tensor(valT[0], dtype=torch.float32)).numpy()
                tft_preds_val[tft_preds_val < 0] = 0
            
            # Create base model predictions dictionary
            base_model_preds = {
                'tcn': tcn_preds_val,
                'lstm': lstm_preds_val,
                'catboost': catboost_preds_val,
                'tft': tft_preds_val
            }
            
            # Generate predictions using the loaded ensemble
            ensemble_predictions = ensemble.predict(base_model_preds)
            
        except Exception as e:
            print(f"Error during inference: {e}")
            import traceback
            traceback.print_exc()
            return
    
        try:
            # Calculate metrics for each department
            results = []
            # Invert scaling
            # Load original scaler
            scaler = joblib.load('scaler_dengue.save')
            preds_original = scaler.inverse_transform(ensemble_predictions)
            truth_original = scaler.inverse_transform(y_val)
            
            for department_idx, department_name in DEP_NAMES.items():
                # Filter rows corresponding to the current department for validation
                department_rows_val = validation_dataframe[validation_dataframe['dep_id'] == department_idx]
                department_rows_train = training_dataframe[training_dataframe['dep_id'] == department_idx]
                
                if department_rows_val.empty or department_rows_train.empty:
                    continue
                
                department_indices_val = department_rows_val.index.tolist()
                department_indices_train = department_rows_train.index.tolist()
                
                # Matching indices for validation data
                matching_indices_val = y_val_indices_df[y_val_indices_df['actual_index'].isin(department_indices_val)].index
                
                if len(matching_indices_val) == 0:
                    continue
                
                # Extract true and predicted values for validation data
                true_dengrate_all_val = truth_original[matching_indices_val, 0]
                true_dengrate_019_val = truth_original[matching_indices_val, 1]
                predicted_dengrate_all_val = preds_original[matching_indices_val, 0]
                predicted_dengrate_019_val = preds_original[matching_indices_val, 1]
                
                # Calculate NRMSE for both DengRate_all and DengRate_019 (validation data)
                nrmse_dengrate_all_val = root_mean_squared_error(true_dengrate_all_val, predicted_dengrate_all_val) / (true_dengrate_all_val.max() - true_dengrate_all_val.min())
                nrmse_dengrate_019_val = root_mean_squared_error(true_dengrate_019_val, predicted_dengrate_019_val) / (true_dengrate_019_val.max() - true_dengrate_019_val.min())
                
                # Calculate MAE for both DengRate_all and DengRate_019 (validation data)
                mae_dengrate_all_val = mean_absolute_error(true_dengrate_all_val, predicted_dengrate_all_val)
                mae_dengrate_019_val = mean_absolute_error(true_dengrate_019_val, predicted_dengrate_019_val)
                
                # Store the results
                results.append({
                    'Department': department_name,
                    'NRMSE 0-19 Training': nrmse_dengrate_019_val,  # Using validation metrics for both
                    'NRMSE All Training': nrmse_dengrate_all_val,
                    'NRMSE 0-19 Validation': nrmse_dengrate_019_val,
                    'NRMSE All Validation': nrmse_dengrate_all_val,
                    'MAE (DengRate_all) Val': mae_dengrate_all_val,
                    'MAE (DengRate_019) Val': mae_dengrate_019_val
                })
            
            # Save results
            results_df = pd.DataFrame(results)
            today = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
            results_path = os.path.join(metrics_path, f'ensemble_results_{today}.csv')
            results_df.to_csv(results_path, index=False)
            print(f"Results saved to {results_path}")
            
            # Print average metrics
            print("\nAverage Metrics:")
            print(f"NRMSE All Validation: {results_df['NRMSE All Validation'].mean():.4f}")
            print(f"NRMSE 0-19 Validation: {results_df['NRMSE 0-19 Validation'].mean():.4f}")
            print(f"MAE (DengRate_all) Val: {results_df['MAE (DengRate_all) Val'].mean():.4f}")
            print(f"MAE (DengRate_019) Val: {results_df['MAE (DengRate_019) Val'].mean():.4f}")
            
        except Exception as e:
            print(f"Error during metrics calculation: {e}")
            import traceback
            traceback.print_exc()

if __name__ == "__main__":
    main() 