import numpy as np
import pandas as pd
import lightgbm as lgb
import optuna  # new dependency
from sklearn.metrics import mean_squared_error
import os
from datetime import datetime
import joblib

class BlendedStackingEnsemble:
    def __init__(self, base_model_weights=None, blend_ratio=0.7):
        """
        Initialize with optional base_model_weights and blend_ratio.
        """
        self.base_model_weights = base_model_weights or {
            'tcn': 0.25, 'lstm': 0.25, 'catboost': 0.25, 'tft': 0.25
        }
        self.blend_ratio = blend_ratio
        self.target_names = ['DengRate_all', 'DengRate_019']
        self.meta_models = {}

    def train_meta_model(self, base_model_preds, y_true):
        """
        Train separate LightGBM meta-models (one per target) with early stopping.
        """
        X = np.column_stack([base_model_preds[n] for n in self.base_model_weights])
        params = {
            'boosting_type': 'gbdt', 'objective': 'regression', 'metric': 'rmse',
            'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.9,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'n_estimators': 1000,
            'verbose': -1
        }
        # from sklearn.model_selection import train_test_split
        # X_tr, X_val, y_tr, y_val = train_test_split(X, y_true, test_size=0.2, random_state=42)

        for idx, target in enumerate(self.target_names):
            model = lgb.LGBMRegressor(**params)
            # Fit on full X, y_true[:, idx]
            model.fit(X, y_true[:, idx])
            self.meta_models[target] = model

    def save_model(self, output_path):
        """
        Save per-target meta-models and base weights.
        """
        today = datetime.now().strftime("%d-%m-%Y-%H-%M-%S")
        model_dir = os.path.join(output_path, f'blended_stacking_{today}')
        os.makedirs(model_dir, exist_ok=True)

        for target, model in self.meta_models.items():
            joblib.dump(model, os.path.join(model_dir, f'meta_{target}.pkl'))
        joblib.dump(self.base_model_weights, os.path.join(model_dir, 'base_weights.joblib'))
        joblib.dump(self.blend_ratio, os.path.join(model_dir, 'blend_ratio.joblib'))
        print(f"Saved models to {model_dir}")

    def load_model(self, model_dir):
        """
        Load per-target meta-models and base weights.
        """
        self.meta_models = {}
        for target in self.target_names:
            path = os.path.join(model_dir, f'meta_{target}.pkl')
            self.meta_models[target] = joblib.load(path)

        self.base_model_weights = joblib.load(os.path.join(model_dir, 'base_weights.joblib'))
        self.blend_ratio = joblib.load(os.path.join(model_dir, 'blend_ratio.joblib'))
        print(f"Loaded models from {model_dir}")

    def predict(self, base_model_preds):
        """
        Generate final blended predictions using current weights & blend_ratio.
        """
        X = np.column_stack([base_model_preds[n] for n in self.base_model_weights])
        meta_preds = np.vstack([self.meta_models[t].predict(X) for t in self.target_names]).T
        linear = sum(base_model_preds[n] * w for n, w in self.base_model_weights.items())
        return self.blend_ratio * meta_preds + (1 - self.blend_ratio) * linear

    def optimize_weights(self, base_model_preds_val, y_val, n_trials=100, prune=True):
        """
        Optimize base_model_weights and blend_ratio via Optuna (TPE sampler).
        """
        model_names = list(self.base_model_weights.keys())

        def objective(trial):
            # Suggest and normalize base weights
            raw = [trial.suggest_float(f'w_{n}', 0.0, 1.0) for n in model_names]
            total = sum(raw) or 1.0
            weights = {n: r/total for n, r in zip(model_names, raw)}
            br = trial.suggest_float('blend_ratio', 0.0, 1.0)

            # Linear blend
            linear = sum(base_model_preds_val[n] * weights[n] for n in model_names)
            # Meta stack
            Xv = np.column_stack([base_model_preds_val[n] for n in model_names])
            meta = np.vstack([self.meta_models[t].predict(Xv) for t in self.target_names]).T
            preds = br * meta + (1 - br) * linear

            # Loss with light regularization
            rmse = mean_squared_error(y_val, preds, squared=False)
            return rmse

        pruner = optuna.pruners.MedianPruner() if prune else optuna.pruners.NopPruner()
        study = optuna.create_study(direction='minimize', sampler=optuna.samplers.TPESampler(seed=42), pruner=pruner)
        study.optimize(objective, n_trials=n_trials)

        best = study.best_trial.params
        raw = [best[f'w_{n}'] for n in model_names]
        total = sum(raw) or 1.0
        self.base_model_weights = {n: r/total for n, r in zip(model_names, raw)}
        self.blend_ratio = best['blend_ratio']

        print('Optimized weights:', self.base_model_weights)
        print('Optimized blend_ratio:', self.blend_ratio)
