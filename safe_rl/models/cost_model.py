import os
import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from joblib import Parallel, delayed

class CostModelEnsemble:
    def __init__(self, n_models=5):
        self.n_models = n_models
        self.models = []
        self.trained = False

    def fit(self, X, y):
        self.models = []
        for i in range(self.n_models):
            model = RandomForestRegressor(n_estimators=20)
            X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2)
            model.fit(X_train, y_train)
            self.models.append(model)
        self.trained = True

    def predict(self, X):
        if not self.models:
            raise RuntimeError("No models loaded")
        preds = np.stack([model.predict(X) for model in self.models])
        mean = np.mean(preds, axis=0)
        std = np.std(preds, axis=0)
        return mean, std

    def safe_action_mask(self, obs_batch, action_batch, epsilon=0.1, quantile=0.95):
        if not self.models:
            raise RuntimeError("No models loaded")
        X = np.hstack([obs_batch, action_batch])
        preds = np.stack(Parallel(n_jobs=-1)(delayed(model.predict)(X) for model in self.models))
        threshold = np.quantile(preds, quantile, axis=0)
        return threshold < epsilon

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        for i, model in enumerate(self.models):
            joblib.dump(model, os.path.join(path, f"cost_model_{i}.joblib"))

    def load(self, path):
        self.models = []
        for i in range(self.n_models):
            model_path = os.path.join(path, f"cost_model_{i}.joblib")
            if os.path.exists(model_path):
                model = joblib.load(model_path)
                self.models.append(model)
        self.trained = len(self.models) > 0
