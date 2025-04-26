import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import time


class SurgePredictionModel:
    """
    Class for training, evaluating, and making predictions with a surge pricing model.
    """
    
    def __init__(self, model_type: str = 'xgboost'):
        """
        Initialize the surge prediction model.
        
        Args:
            model_type: Type of model to use ('xgboost' or 'random_forest')
        """
        self.model_type = model_type.lower()
        self.model = None
        self.feature_names = None
        self.metrics = {}
        self.training_time = None
    
    def train(self, X: np.ndarray, y: np.ndarray, feature_names: List[str],
              test_size: float = 0.2, random_state: int = 42) -> Dict[str, float]:
        """
        Train the surge prediction model.
        
        Args:
            X: Feature matrix
            y: Target values (surge multipliers)
            feature_names: Names of the features
            test_size: Proportion of data to use for testing
            random_state: Random seed for reproducibility
            
        Returns:
            Dictionary of evaluation metrics
        """
        # Start timing
        start_time = time.time()
        
        # Store feature names
        self.feature_names = feature_names
        
        # Split data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        print(f"Training {self.model_type} model on {X_train.shape[0]} samples...")
        
        # Create and train the model
        if self.model_type == 'xgboost':
            self.model = xgb.XGBRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=6,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.01,
                reg_lambda=1.0,
                random_state=random_state,
                n_jobs=-1
            )
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                max_depth=15,
                min_samples_split=5,
                min_samples_leaf=2,
                max_features='sqrt',
                random_state=random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        # Train the model
        self.model.fit(X_train, y_train)
        
        # Record training time
        self.training_time = time.time() - start_time
        
        # Evaluate the model
        self.metrics = self._evaluate_model(X_train, y_train, X_test, y_test)
        
        return self.metrics
    
    def _evaluate_model(self, X_train, y_train, X_test, y_test) -> Dict[str, float]:
        """Evaluate the model and return performance metrics."""
        # Make predictions on test data
        y_pred = self.model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Cross-validation on training data
        cv_scores = cross_val_score(
            self.model, X_train, y_train, 
            cv=5, scoring='neg_mean_squared_error'
        )
        cv_rmse = np.sqrt(-cv_scores.mean())
        
        # Calculate baseline: always predict mean
        y_mean = np.full_like(y_test, y_train.mean())
        baseline_mse = mean_squared_error(y_test, y_mean)
        improvement_over_baseline = (1 - mse / baseline_mse) * 100
        
        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2,
            'cv_rmse': cv_rmse,
            'baseline_mse': baseline_mse,
            'improvement_over_baseline': improvement_over_baseline,
            'training_time': self.training_time
        }
        
        # Print metrics
        print("\nModel Evaluation Metrics:")
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
        print(f"Mean Absolute Error (MAE): {mae:.4f}")
        print(f"R² Score: {r2:.4f}")
        print(f"Cross-validation RMSE: {cv_rmse:.4f}")
        print(f"Improvement over baseline: {improvement_over_baseline:.2f}%")
        print(f"Training time: {self.training_time:.2f} seconds")
        
        return metrics
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Make surge multiplier predictions.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted surge multipliers
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        return self.model.predict(X)
    
    def predict_single(self, X_single: np.ndarray) -> float:
        """
        Make a surge multiplier prediction for a single example.
        
        Args:
            X_single: Feature vector for a single example
            
        Returns:
            Predicted surge multiplier
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Ensure X_single is 2D
        if X_single.ndim == 1:
            X_single = X_single.reshape(1, -1)
        
        return float(self.model.predict(X_single)[0])
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance from the model.
        
        Returns:
            DataFrame with feature names and their importance
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        if self.feature_names is None:
            raise ValueError("Feature names not available.")
        
        # Get feature importance
        importance = self.model.feature_importances_
        
        # Create DataFrame
        importance_df = pd.DataFrame({
            'Feature': self.feature_names,
            'Importance': importance
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        return importance_df
    
    def plot_feature_importance(self, top_n: int = 15, figsize: Tuple[int, int] = (10, 8)) -> None:
        """
        Plot feature importance.
        
        Args:
            top_n: Number of top features to display
            figsize: Figure size (width, height)
        """
        importance_df = self.get_feature_importance()
        
        # Select top N features
        importance_df = importance_df.head(top_n)
        
        # Plot
        plt.figure(figsize=figsize)
        sns.barplot(x='Importance', y='Feature', data=importance_df)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        os.makedirs('models', exist_ok=True)
        plt.savefig(os.path.join('models', 'feature_importance.png'))
        
        plt.show()
    
    def save_model(self, filepath: str) -> None:
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        if self.model is None:
            raise ValueError("Model has not been trained yet. Call train() first.")
        
        # Create model data dictionary
        model_data = {
            'model': self.model,
            'feature_names': self.feature_names,
            'metrics': self.metrics,
            'model_type': self.model_type,
            'training_time': self.training_time
        }
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model data
        joblib.dump(model_data, filepath)
        print(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str) -> None:
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
        """
        # Load model data
        model_data = joblib.load(filepath)
        
        # Extract model components
        self.model = model_data['model']
        self.feature_names = model_data['feature_names']
        self.metrics = model_data['metrics']
        self.model_type = model_data['model_type']
        self.training_time = model_data['training_time']
        
        print(f"Model loaded from {filepath}")
        
        # Print metrics
        if self.metrics:
            print("\nModel Metrics:")
            print(f"Root Mean Squared Error (RMSE): {self.metrics.get('rmse', 'N/A')}")
            print(f"Mean Absolute Error (MAE): {self.metrics.get('mae', 'N/A')}")
            print(f"R² Score: {self.metrics.get('r2', 'N/A')}")


def compare_models(X: np.ndarray, y: np.ndarray, feature_names: List[str],
                  test_size: float = 0.2, random_state: int = 42) -> Dict[str, Dict[str, float]]:
    """
    Compare different models for surge prediction.
    
    Args:
        X: Feature matrix
        y: Target values (surge multipliers)
        feature_names: Names of the features
        test_size: Proportion of data to use for testing
        random_state: Random seed for reproducibility
        
    Returns:
        Dictionary of model names and their metrics
    """
    # Models to compare
    models = {
        'xgboost': SurgePredictionModel('xgboost'),
        'random_forest': SurgePredictionModel('random_forest')
    }
    
    # Results dictionary
    results = {}
    
    # Train and evaluate each model
    for name, model in models.items():
        print(f"\n{'-'*40}")
        print(f"Training {name} model")
        print(f"{'-'*40}")
        
        metrics = model.train(X, y, feature_names, test_size, random_state)
        results[name] = metrics
    
    # Compare results
    print("\nModel Comparison:")
    for name, metrics in results.items():
        print(f"\n{name.upper()}:")
        print(f"RMSE: {metrics['rmse']:.4f}")
        print(f"MAE: {metrics['mae']:.4f}")
        print(f"R²: {metrics['r2']:.4f}")
    
    # Return best model based on RMSE
    best_model = min(results.items(), key=lambda x: x[1]['rmse'])[0]
    print(f"\nBest model based on RMSE: {best_model.upper()}")
    
    return results


if __name__ == "__main__":
    # Example usage
    from data_preprocessing import DataPreprocessor
    from feature_engineering import FeatureEngineer
    
    # Paths
    rides_path = os.path.join("data", "cab_rides.csv")
    weather_path = os.path.join("data", "weather.csv")
    
    # Load and preprocess data
    preprocessor = DataPreprocessor(rides_path, weather_path)
    preprocessor.load_data()
    preprocessor.clean_rides_data()
    preprocessor.clean_weather_data()
    merged_data = preprocessor.merge_datasets()
    
    # Engineer features
    engineer = FeatureEngineer()
    features_df = engineer.engineer_features(merged_data)
    
    # Prepare features and target
    X, feature_names = engineer.prepare_features_for_model(features_df)
    y = merged_data['surge_multiplier'].values
    
    # Train model
    model = SurgePredictionModel('xgboost')
    metrics = model.train(X, y, feature_names)
    
    # Plot feature importance
    model.plot_feature_importance()
    
    # Save model
    model.save_model(os.path.join("models", "surge_model.joblib")) 