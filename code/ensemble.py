import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import mean_squared_error, mean_absolute_error
import os
from datetime import datetime
import joblib

class BlendedStackingEnsemble:
    def __init__(self, base_model_weights=None):
        """
        Initialize the blended stacking ensemble with per-target LightGBM meta-models.
        """
        self.meta_models = {}  # {'DengRate_all': model, 'DengRate_019': model}
        self.base_model_weights = base_model_weights or {
            'tcn': 0.25,
            'lstm': 0.25,
            'catboost': 0.25,
            'tft': 0.25
        }
        self.target_names = ['DengRate_all', 'DengRate_019']

    def train_meta_model(self, base_model_preds, y_true):
        """
        Train separate LightGBM meta-models (one per target) with early stopping.
        """
        X = np.column_stack(list(base_model_preds.values()))
        # LightGBM parameters
        lgb_params = {
            'boosting_type': 'gbdt',
            'objective': 'regression',
            'metric': 'rmse',
            'num_leaves': 31,
            'learning_rate': 0.05,
            'feature_fraction': 0.9,
            'bagging_fraction': 0.8,
            'bagging_freq': 5,
            'n_estimators': 1000,
            'verbose': -1
        }
        # Split for early stopping
        from sklearn.model_selection import train_test_split
        X_tr, X_val, y_tr, y_val = train_test_split(
            X, y_true, test_size=0.2, random_state=42
        )
        for idx, target in enumerate(self.target_names):
            model = lgb.LGBMRegressor(**lgb_params)
            model.fit(
                X_tr, y_tr[:, idx],
                eval_set=[(X_val, y_val[:, idx])],
                eval_metric='rmse',
                callbacks=[lgb.early_stopping(stopping_rounds=50)],
                verbose=False
            )
            self.meta_models[target] = model

    def save_model(self, output_path):
        """
        Save per-target meta-models and blend weights.
        """
        today = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        model_dir = os.path.join(output_path, f'blended_stacking_{today}')
        os.makedirs(model_dir, exist_ok=True)
        for target, model in self.meta_models.items():
            model.booster_.save_model(os.path.join(model_dir, f'meta_{target}.txt'))
        joblib.dump(self.base_model_weights, os.path.join(model_dir, 'base_weights.joblib'))

    def load_model(self, model_dir):
        """
        Load per-target meta-models and blend weights.
        """
        self.meta_models = {}
        for target in self.target_names:
            # Load the raw Booster
            booster = lgb.Booster(model_file=os.path.join(model_dir, f'meta_{target}.txt'))
            self.meta_models[target] = booster
        # Load blend weights
        self.base_model_weights = joblib.load(os.path.join(model_dir, 'base_weights.joblib'))

    def predict(self, base_model_preds):
        """
        Generate blended predictions for each target using raw boosters.
        """
        X = np.column_stack(list(base_model_preds.values()))
        # Meta-model predictions via Booster.predict
        meta_preds = np.vstack([
            self.meta_models[target].predict(X)
            for target in self.target_names
        ]).T
        # Weighted average of base model predictions
        weighted = np.zeros_like(meta_preds)
        for name, preds in base_model_preds.items():
            weighted += preds * self.base_model_weights[name]
        return 0.7 * meta_preds + 0.3 * weighted
