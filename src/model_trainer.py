import pandas as pd
import numpy as np
import os
import joblib
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score, accuracy_score, precision_score, recall_score, f1_score, classification_report
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

class ModelTrainer:
    def __init__(self, models_dir='models'):
        """
        Initialize ModelTrainer
        """
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        # Create a subdirectory for plots
        self.plots_dir = os.path.join(models_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def train_model(self, model, X_train, y_train, model_name=None, hyperparams=None):
        """
        Train a model with optional hyperparameter tuning
        """
        if hyperparams is not None:
            # Perform grid search for hyperparameter tuning
            grid_search = GridSearchCV(
                model, 
                hyperparams, 
                cv=5, 
                scoring='neg_mean_squared_error' if y_train.dtype.kind in 'fi' else 'f1_weighted',
                n_jobs=-1,
                verbose=1
            )
            grid_search.fit(X_train, y_train)
            
            # Get the best model
            best_model = grid_search.best_estimator_
            print(f"Best parameters: {grid_search.best_params_}")
            
            # Save the grid search results
            results_df = pd.DataFrame(grid_search.cv_results_)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_df.to_csv(os.path.join(self.models_dir, f'gridsearch_results_{timestamp}.csv'), index=False)
            
            return best_model
        else:
            # Train the model without hyperparameter tuning
            model.fit(X_train, y_train)
            return model
    
    def evaluate_regression_model(self, model, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate a regression model
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        # Print metrics
        print(f"Test set evaluation:")
        print(f"MSE: {mse:.4f}")
        print(f"RMSE: {rmse:.4f}")
        print(f"MAE: {mae:.4f}")
        print(f"R²: {r2:.4f}")
        
        # Also evaluate on training set if provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_mse = mean_squared_error(y_train, y_train_pred)
            train_rmse = np.sqrt(train_mse)
            train_r2 = r2_score(y_train, y_train_pred)
            
            print(f"\nTrain set evaluation:")
            print(f"MSE: {train_mse:.4f}")
            print(f"RMSE: {train_rmse:.4f}")
            print(f"R²: {train_r2:.4f}")
            
            # Check for overfitting
            if train_rmse < rmse * 0.7:
                print("\nWarning: Model might be overfitting. Train RMSE is significantly lower than Test RMSE.")
        
        # Plot actual vs predicted values
        plt.figure(figsize=(10, 6))
        plt.scatter(y_test, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_test.min(), y_pred.min())
        max_val = max(y_test.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Actual vs Predicted Values')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'regression_eval_{timestamp}.png'))
        plt.close()
        
        # Return metrics
        return {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
    
    def evaluate_classification_model(self, model, X_test, y_test, X_train=None, y_train=None):
        """
        Evaluate a classification model
        """
        # Make predictions
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        
        # Check if binary or multiclass
        if len(np.unique(y_test)) <= 2:
            # Binary classification
            precision = precision_score(y_test, y_pred)
            recall = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            print(f"Test set evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Precision: {precision:.4f}")
            print(f"Recall: {recall:.4f}")
            print(f"F1 Score: {f1:.4f}")
        else:
            # Multiclass classification
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            f1 = f1_score(y_test, y_pred, average='weighted')
            
            print(f"Test set evaluation:")
            print(f"Accuracy: {accuracy:.4f}")
            print(f"Weighted Precision: {precision:.4f}")
            print(f"Weighted Recall: {recall:.4f}")
            print(f"Weighted F1 Score: {f1:.4f}")
        
        # Print classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Also evaluate on training set if provided
        if X_train is not None and y_train is not None:
            y_train_pred = model.predict(X_train)
            train_accuracy = accuracy_score(y_train, y_train_pred)
            
            print(f"\nTrain set accuracy: {train_accuracy:.4f}")
            
            # Check for overfitting
            if train_accuracy > accuracy * 1.3:
                print("\nWarning: Model might be overfitting. Train accuracy is significantly higher than Test accuracy.")
        
        # Plot confusion matrix
        plt.figure(figsize=(10, 8))
        cm = pd.crosstab(y_test, y_pred, rownames=['Actual'], colnames=['Predicted'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'confusion_matrix_{timestamp}.png'))
        plt.close()
        
        # Return metrics
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
    
    def save_model(self, model, model_name):
        """
        Save the trained model
        """
        # Add timestamp to model name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"{model_name}_{timestamp}.joblib"
        filepath = os.path.join(self.models_dir, filename)
        
        # Save the model
        joblib.dump(model, filepath)
        print(f"Model saved to {filepath}")
        
        return filepath
    
    def load_model(self, model_path):
        """
        Load a saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        # Load the model
        model = joblib.load(model_path)
        return model
    
    def plot_feature_importance(self, model, feature_names, top_n=20):
        """
        Plot feature importance for tree-based models
        """
        # Check if model has feature_importances_ attribute
        if not hasattr(model, 'feature_importances_'):
            print("This model doesn't have feature_importances_ attribute. Skipping feature importance plot.")
            return
        
        # Get feature importances
        importances = model.feature_importances_
        
        # Create DataFrame for plotting
        feature_importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        })
        
        # Sort by importance
        feature_importance_df = feature_importance_df.sort_values('Importance', ascending=False)
        
        # Plot top N features
        plt.figure(figsize=(12, 8))
        top_features = feature_importance_df.head(top_n)
        sns.barplot(x='Importance', y='Feature', data=top_features)
        plt.title(f'Top {top_n} Feature Importance')
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'feature_importance_{timestamp}.png'))
        plt.close()
        
        return feature_importance_df
    
    def plot_learning_curve(self, model, X_train, y_train, cv=5):
        """
        Plot learning curve to diagnose overfitting/underfitting
        """
        from sklearn.model_selection import learning_curve
        
        # Calculate learning curve
        train_sizes, train_scores, test_scores = learning_curve(
            model, X_train, y_train, cv=cv, n_jobs=-1, 
            train_sizes=np.linspace(0.1, 1.0, 10)
        )
        
        # Calculate mean and std for training set scores
        train_mean = np.mean(train_scores, axis=1)
        train_std = np.std(train_scores, axis=1)
        
        # Calculate mean and std for test set scores
        test_mean = np.mean(test_scores, axis=1)
        test_std = np.std(test_scores, axis=1)
        
        # Plot learning curve
        plt.figure(figsize=(12, 8))
        plt.plot(train_sizes, train_mean, label="Training score", color="blue")
        plt.fill_between(train_sizes, train_mean - train_std, train_mean + train_std, color="blue", alpha=0.1)
        plt.plot(train_sizes, test_mean, label="Cross-validation score", color="green")
        plt.fill_between(train_sizes, test_mean - test_std, test_mean + test_std, color="green", alpha=0.1)
        
        plt.xlabel("Training set size")
        plt.ylabel("Score")
        plt.title("Learning Curve")
        plt.legend(loc="best")
        plt.grid(True)
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'learning_curve_{timestamp}.png'))
        plt.close()
    
    def full_training_pipeline(self, model, X_train, y_train, X_test, y_test, model_name,
                              hyperparams=None, is_regression=True, feature_names=None):
        """
        Run the full training and evaluation pipeline
        """
        # Train the model
        trained_model = self.train_model(model, X_train, y_train, hyperparams=hyperparams)
        
        # Evaluate the model
        if is_regression:
            metrics = self.evaluate_regression_model(trained_model, X_test, y_test, X_train, y_train)
        else:
            metrics = self.evaluate_classification_model(trained_model, X_test, y_test, X_train, y_train)
        
        # Plot feature importance if feature names are provided
        if feature_names is not None:
            self.plot_feature_importance(trained_model, feature_names)
        
        # Plot learning curve
        self.plot_learning_curve(trained_model, X_train, y_train)
        
        # Save the model
        model_path = self.save_model(trained_model, model_name)
        
        return trained_model, metrics, model_path 