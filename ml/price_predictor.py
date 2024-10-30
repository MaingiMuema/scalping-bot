import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler
import os
import joblib
import logging
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RandomizedSearchCV

logger = logging.getLogger(__name__)

class PricePredictor:
    def __init__(self, model_path=None):
        self.models = {
            'rf': RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42),
            'gb': GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42),
            'lr': LinearRegression(),
            'svr': SVR(kernel='rbf')
        }
        self.best_model = None
        self.scaler = StandardScaler()
        self.model_path = model_path
        self.is_trained = False

    def set_params(self, **params):
        if 'lgbm' in params:
            self.lgbm.set_params(**params['lgbm'])
        if 'rf' in params:
            self.rf.set_params(**params['rf'])

    def is_model_trained(self):
        return self.is_trained

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            self.model, self.scaler = joblib.load(self.model_path)
            self.is_trained = True
            logger.info(f"Loaded model from {self.model_path}")

    def train(self, X: np.array, y: np.array):
        X_scaled = self.scaler.fit_transform(X)
        
        best_score = float('-inf')
        for name, model in self.models.items():
            score = np.mean(cross_val_score(model, X_scaled, y, cv=5, scoring='neg_mean_squared_error'))
            if score > best_score:
                best_score = score
                self.best_model = model
        
        self.best_model.fit(X_scaled, y)
        self.is_trained = True
        if self.model_path:
            self.save_model()
        logger.info(f"Model trained successfully. Best model: {type(self.best_model).__name__}")

    def predict(self, X: np.array) -> np.array:
        if not self.is_trained:
            raise ValueError("Model is not trained yet")
        X_scaled = self.scaler.transform(X)
        return self.best_model.predict(X_scaled)

    def save_model(self):
        if self.model_path:
            joblib.dump((self.model, self.scaler), self.model_path)
            logger.info(f"Saved model to {self.model_path}")

    def load_model(self):
        if self.model_path and os.path.exists(self.model_path):
            self.model, self.scaler = joblib.load(self.model_path)
            self.is_trained = True
            logger.info(f"Loaded model from {self.model_path}")

    def tune_hyperparameters(self, X: np.array, y: np.array):
        X_scaled = self.scaler.fit_transform(X)
        
        param_distributions = {
            'n_estimators': [50, 100, 200, 300],
            'max_depth': [5, 10, 15, 20, None],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
        
        random_search = RandomizedSearchCV(
            self.best_model, param_distributions, n_iter=20, cv=5, 
            scoring='neg_mean_squared_error', n_jobs=-1, random_state=42
        )
        
        random_search.fit(X_scaled, y)
        self.best_model = random_search.best_estimator_
        logger.info(f"Hyperparameters tuned. Best params: {random_search.best_params_}")
