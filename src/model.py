import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
import xgboost as xgb
import joblib
from typing import Tuple, Union, Dict, List, Optional
import matplotlib.pyplot as plt

class SurgePredictor:
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the surge predictor model.
        
        Args:
            model_type: Either 'xgboost', 'random_forest', or 'gradient_boosting'
        """
        self.model_type = model_type
        self.model = None
        self.feature_importance = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.cross_val_scores = None
        
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2,
              tune_hyperparameters: bool = False) -> Dict[str, float]:
        """
        Train the model and return performance metrics.
        
        Args:
            X: Feature matrix
            y: Target variable (surge multiplier)
            test_size: Proportion of data to use for testing
            tune_hyperparameters: Whether to perform hyperparameter tuning
            
        Returns:
            Dictionary containing performance metrics
        """
        # Store feature names for later use
        self.feature_names = X.columns.tolist()
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )
        
        # Scale the data
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Initialize and train the model
        if tune_hyperparameters:
            self.model = self._tune_hyperparameters(X_train_scaled, y_train)
        else:
            self.model = self._get_default_model()
        
        # Train the model
        self.model.fit(X_train_scaled, y_train)
        
        # Compute cross-validation scores
        self.cross_val_scores = cross_val_score(
            self.model, X_train_scaled, y_train, cv=5, scoring='neg_mean_squared_error'
        )
        
        # Get feature importance
        self._compute_feature_importance(X)
        
        # Make predictions and calculate metrics
        y_pred = self.model.predict(X_test_scaled)
        metrics = {
            'mse': mean_squared_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'cross_val_rmse': np.sqrt(-np.mean(self.cross_val_scores))
        }
        
        return metrics
    
    def _get_default_model(self):
        """Return the default model based on model_type."""
        if self.model_type == 'xgboost':
            return xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42,
                n_jobs=-1
            )
        elif self.model_type == 'gradient_boosting':
            return GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=5,
                random_state=42
            )
        else:  # random_forest
            return RandomForestRegressor(
                n_estimators=100,
                max_depth=10,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
    
    def _tune_hyperparameters(self, X_train: np.ndarray, y_train: pd.Series):
        """Tune model hyperparameters using grid search."""
        if self.model_type == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0],
                'colsample_bytree': [0.8, 1.0]
            }
            model = xgb.XGBRegressor(random_state=42)
        
        elif self.model_type == 'gradient_boosting':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'learning_rate': [0.01, 0.1, 0.2],
                'max_depth': [3, 5, 7],
                'subsample': [0.8, 1.0]
            }
            model = GradientBoostingRegressor(random_state=42)
        
        else:  # random_forest
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            model = RandomForestRegressor(random_state=42)
        
        # Run grid search
        grid_search = GridSearchCV(
            model, param_grid, cv=3, scoring='neg_mean_squared_error', 
            n_jobs=-1, verbose=1, return_train_score=True
        )
        grid_search.fit(X_train, y_train)
        
        # Get best model
        return grid_search.best_estimator_
    
    def _compute_feature_importance(self, X: pd.DataFrame):
        """Compute and store feature importance."""
        if hasattr(self.model, 'feature_importances_'):
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': self.model.feature_importances_
            }).sort_values('importance', ascending=False)
        else:
            # For models that don't have feature_importances_ attribute,
            # use permutation importance or another method
            self.feature_importance = pd.DataFrame({
                'feature': X.columns,
                'importance': np.ones(len(X.columns)) / len(X.columns)
            })
    
    def predict(self, X: pd.DataFrame) -> float:
        """
        Make predictions for new data.
        
        Args:
            X: Feature matrix for prediction
            
        Returns:
            Predicted surge multiplier
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Scale the input data
        X_scaled = self.scaler.transform(X)
        
        # Make prediction
        pred = float(self.model.predict(X_scaled)[0])
        
        # Ensure prediction is reasonable (non-negative)
        return max(1.0, pred)
    
    def evaluate_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """
        Evaluate model on test data and return detailed metrics.
        
        Args:
            X: Test feature matrix
            y: Test target variable
            
        Returns:
            Dictionary of evaluation metrics
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet!")
        
        # Scale the data
        X_scaled = self.scaler.transform(X)
        
        # Make predictions
        y_pred = self.model.predict(X_scaled)
        
        # Calculate metrics
        metrics = {
            'mse': mean_squared_error(y, y_pred),
            'rmse': np.sqrt(mean_squared_error(y, y_pred)),
            'mae': mean_absolute_error(y, y_pred),
            'r2': r2_score(y, y_pred)
        }
        
        # Add cross validation score if available
        if self.cross_val_scores is not None:
            metrics['cross_val_rmse'] = np.sqrt(-np.mean(self.cross_val_scores))
        
        return metrics
    
    def save_model(self, path: str):
        """Save the trained model to disk."""
        if self.model is None:
            raise ValueError("No model to save!")
        
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'feature_importance': self.feature_importance,
            'model_type': self.model_type,
            'cross_val_scores': self.cross_val_scores
        }
        
        joblib.dump(model_data, path)
    
    def load_model(self, path: str):
        """Load a trained model from disk."""
        model_data = joblib.load(path)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.feature_importance = model_data['feature_importance']
        self.model_type = model_data['model_type']
        self.cross_val_scores = model_data['cross_val_scores']
    
    def get_feature_importance(self) -> pd.DataFrame:
        """Return feature importance scores."""
        if self.feature_importance is None:
            raise ValueError("Model has not been trained yet!")
        
        return self.feature_importance
    
    def analyze_weather_effect(self, df: pd.DataFrame) -> Dict[str, float]:
        """
        Analyze the effect of weather conditions on surge pricing.
        
        Args:
            df: DataFrame containing merged ride and weather data
            
        Returns:
            Dictionary mapping weather conditions to their impact on surge
        """
        # Check if precipitation_type column exists
        if 'precipitation_type' not in df.columns or 'surge_multiplier' not in df.columns:
            return {}
        
        # Group by precipitation type and calculate mean surge
        weather_impact = df.groupby('precipitation_type')['surge_multiplier'].mean().to_dict()
        
        # Calculate relative impact (compared to clear weather)
        clear_surge = weather_impact.get('clear', 1.0)
        relative_impact = {
            weather: round((surge / clear_surge - 1) * 100, 1)
            for weather, surge in weather_impact.items()
        }
        
        return relative_impact
    
    def analyze_time_patterns(self, df: pd.DataFrame) -> Dict[str, Dict[int, float]]:
        """
        Analyze surge patterns by hour and day of week.
        
        Args:
            df: DataFrame containing ride data with timestamp
            
        Returns:
            Dictionary with hourly and daily surge patterns
        """
        if 'hour' not in df.columns or 'day_of_week' not in df.columns:
            return {}
        
        # Get hourly patterns
        hourly = df.groupby('hour')['surge_multiplier'].mean().to_dict()
        
        # Get daily patterns
        daily = df.groupby('day_of_week')['surge_multiplier'].mean().to_dict()
        
        # Map day numbers to names
        day_names = {
            0: 'Monday', 1: 'Tuesday', 2: 'Wednesday',
            3: 'Thursday', 4: 'Friday', 5: 'Saturday', 6: 'Sunday'
        }
        daily_named = {day_names[day]: surge for day, surge in daily.items()}
        
        return {
            'hourly': hourly,
            'daily': daily_named
        }
    
    def find_optimal_time(self, df: pd.DataFrame, 
                         day_of_week: int, 
                         hour_range: Tuple[int, int] = (0, 23)) -> int:
        """
        Find the optimal time with lowest surge on a given day.
        
        Args:
            df: DataFrame containing ride data
            day_of_week: Day of week (0=Monday, 6=Sunday)
            hour_range: Range of hours to consider
            
        Returns:
            Hour with lowest average surge multiplier
        """
        # Filter data for the specified day and hour range
        mask = (df['day_of_week'] == day_of_week) & \
               (df['hour'] >= hour_range[0]) & \
               (df['hour'] <= hour_range[1])
        
        filtered_df = df[mask]
        
        # Group by hour and find hour with minimum surge
        if len(filtered_df) > 0:
            hourly_surge = filtered_df.groupby('hour')['surge_multiplier'].mean()
            return int(hourly_surge.idxmin())
        else:
            # Return middle of range if no data available
            return (hour_range[0] + hour_range[1]) // 2 