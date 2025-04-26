import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import roc_curve, auc, precision_recall_curve, confusion_matrix
from sklearn.metrics import classification_report
from sklearn.inspection import permutation_importance
import joblib
from datetime import datetime

class ModelEvaluator:
    def __init__(self, models_dir='models', results_dir='evaluation_results'):
        """
        Initialize ModelEvaluator
        """
        self.models_dir = models_dir
        self.results_dir = results_dir
        
        # Create directories if they don't exist
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(results_dir, exist_ok=True)
        
        # Create plots directory
        self.plots_dir = os.path.join(results_dir, 'plots')
        os.makedirs(self.plots_dir, exist_ok=True)
    
    def load_model(self, model_path):
        """
        Load a saved model
        """
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file {model_path} not found")
        
        return joblib.load(model_path)
    
    def evaluate_regression_model(self, model, X, y, dataset_name="test"):
        """
        Evaluate a regression model and return metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Calculate metrics
        mse = mean_squared_error(y, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        
        # Store metrics
        metrics = {
            'dataset': dataset_name,
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        
        return metrics, y_pred
    
    def evaluate_classification_model(self, model, X, y, dataset_name="test"):
        """
        Evaluate a classification model and return metrics
        """
        # Make predictions
        y_pred = model.predict(X)
        
        # Get probabilities if available
        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(X)
        else:
            y_prob = None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        
        # Check if binary or multiclass
        if len(np.unique(y)) <= 2:
            # Binary classification
            precision = precision_score(y, y_pred, zero_division=0)
            recall = recall_score(y, y_pred, zero_division=0)
            f1 = f1_score(y, y_pred, zero_division=0)
            
            avg_type = 'binary'
        else:
            # Multiclass classification
            precision = precision_score(y, y_pred, average='weighted', zero_division=0)
            recall = recall_score(y, y_pred, average='weighted', zero_division=0)
            f1 = f1_score(y, y_pred, average='weighted', zero_division=0)
            
            avg_type = 'weighted'
        
        # Store metrics
        metrics = {
            'dataset': dataset_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'avg_type': avg_type
        }
        
        return metrics, y_pred, y_prob
    
    def plot_residuals(self, y_true, y_pred, dataset_name="test"):
        """
        Plot residuals for regression models
        """
        residuals = y_true - y_pred
        
        # Create figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
        
        # Residuals vs Predicted Values
        ax1.scatter(y_pred, residuals, alpha=0.5)
        ax1.axhline(y=0, color='r', linestyle='--')
        ax1.set_xlabel('Predicted Values')
        ax1.set_ylabel('Residuals')
        ax1.set_title('Residuals vs Predicted Values')
        
        # Histogram of Residuals
        ax2.hist(residuals, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(x=0, color='r', linestyle='--')
        ax2.set_xlabel('Residuals')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Distribution of Residuals')
        
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'residuals_{dataset_name}_{timestamp}.png'))
        plt.close()
    
    def plot_regression_results(self, y_true, y_pred, dataset_name="test"):
        """
        Plot actual vs predicted values for regression models
        """
        plt.figure(figsize=(10, 6))
        plt.scatter(y_true, y_pred, alpha=0.5)
        
        # Add perfect prediction line
        min_val = min(y_true.min(), y_pred.min())
        max_val = max(y_true.max(), y_pred.max())
        plt.plot([min_val, max_val], [min_val, max_val], 'r--')
        
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title(f'Actual vs Predicted Values ({dataset_name} set)')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'regression_results_{dataset_name}_{timestamp}.png'))
        plt.close()
    
    def plot_confusion_matrix(self, y_true, y_pred, class_names=None, dataset_name="test"):
        """
        Plot confusion matrix for classification models
        """
        cm = confusion_matrix(y_true, y_pred)
        
        plt.figure(figsize=(10, 8))
        
        if class_names is None:
            class_names = [str(i) for i in range(len(np.unique(y_true)))]
            
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=class_names, yticklabels=class_names)
        
        plt.xlabel('Predicted Labels')
        plt.ylabel('True Labels')
        plt.title(f'Confusion Matrix ({dataset_name} set)')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'confusion_matrix_{dataset_name}_{timestamp}.png'))
        plt.close()
    
    def plot_roc_curve(self, y_true, y_prob, dataset_name="test"):
        """
        Plot ROC curve for binary classification models
        """
        if y_prob is None:
            print("Probability predictions not available. Skipping ROC curve.")
            return
        
        # Check if binary classification
        if len(np.unique(y_true)) > 2:
            print("ROC curve only available for binary classification. Skipping.")
            return
        
        # Get the probabilities for the positive class
        if y_prob.shape[1] == 2:  # Binary classification with 2 columns
            y_prob = y_prob[:, 1]
        
        # Calculate ROC curve
        fpr, tpr, thresholds = roc_curve(y_true, y_prob)
        roc_auc = auc(fpr, tpr)
        
        # Plot ROC curve
        plt.figure(figsize=(10, 8))
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'Receiver Operating Characteristic ({dataset_name} set)')
        plt.legend(loc="lower right")
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'roc_curve_{dataset_name}_{timestamp}.png'))
        plt.close()
    
    def plot_precision_recall_curve(self, y_true, y_prob, dataset_name="test"):
        """
        Plot Precision-Recall curve for binary classification models
        """
        if y_prob is None:
            print("Probability predictions not available. Skipping Precision-Recall curve.")
            return
        
        # Check if binary classification
        if len(np.unique(y_true)) > 2:
            print("Precision-Recall curve only available for binary classification. Skipping.")
            return
        
        # Get the probabilities for the positive class
        if y_prob.shape[1] == 2:  # Binary classification with 2 columns
            y_prob = y_prob[:, 1]
        
        # Calculate Precision-Recall curve
        precision, recall, thresholds = precision_recall_curve(y_true, y_prob)
        pr_auc = auc(recall, precision)
        
        # Plot Precision-Recall curve
        plt.figure(figsize=(10, 8))
        plt.plot(recall, precision, color='blue', lw=2, label=f'PR curve (area = {pr_auc:.2f})')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title(f'Precision-Recall Curve ({dataset_name} set)')
        plt.legend(loc="best")
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'pr_curve_{dataset_name}_{timestamp}.png'))
        plt.close()
    
    def plot_permutation_importance(self, model, X, y, feature_names=None, n_repeats=10, 
                                   random_state=42, dataset_name="test"):
        """
        Plot permutation feature importance for any model
        """
        # Calculate permutation importance
        r = permutation_importance(model, X, y, n_repeats=n_repeats, random_state=random_state)
        
        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(X.shape[1])]
        
        # Create a DataFrame for easier sorting
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': r.importances_mean,
            'Std': r.importances_std
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('Importance', ascending=False)
        
        # Plot top 20 features or all if less than 20
        n_features = min(20, len(importance_df))
        top_features = importance_df.head(n_features)
        
        plt.figure(figsize=(12, 8))
        plt.barh(range(n_features), top_features['Importance'], align='center', alpha=0.8)
        plt.yticks(range(n_features), top_features['Feature'])
        plt.xlabel('Importance')
        plt.title(f'Permutation Feature Importance ({dataset_name} set)')
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'permutation_importance_{dataset_name}_{timestamp}.png'))
        plt.close()
        
        return importance_df
    
    def save_evaluation_report(self, metrics, model_info, dataset_info, filename=None):
        """
        Save evaluation report to CSV
        """
        # Create a DataFrame for the report
        report_data = {**metrics, **model_info, **dataset_info}
        report_df = pd.DataFrame([report_data])
        
        # Generate filename if not provided
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"evaluation_report_{timestamp}.csv"
        
        # Save to CSV
        filepath = os.path.join(self.results_dir, filename)
        report_df.to_csv(filepath, index=False)
        
        print(f"Evaluation report saved to {filepath}")
        
        return filepath
    
    def compare_models(self, models_metrics, model_names, metric_name, is_higher_better=True):
        """
        Compare models based on a specific metric
        """
        # Prepare data for plotting
        data = {'Model': model_names, 'Metric': [m[metric_name] for m in models_metrics]}
        df = pd.DataFrame(data)
        
        # Sort based on metric (ascending or descending)
        df = df.sort_values('Metric', ascending=not is_higher_better)
        
        # Plot comparison
        plt.figure(figsize=(12, 6))
        bars = plt.barh(df['Model'], df['Metric'], color='skyblue')
        
        # Add metric values on bars
        for bar in bars:
            width = bar.get_width()
            plt.text(width + (width * 0.01), bar.get_y() + bar.get_height()/2, 
                    f'{width:.4f}', va='center')
        
        plt.xlabel(metric_name)
        plt.title(f'Model Comparison by {metric_name}')
        plt.tight_layout()
        
        # Save the plot
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plt.savefig(os.path.join(self.plots_dir, f'model_comparison_{metric_name}_{timestamp}.png'))
        plt.close()
    
    def evaluate_model_pipeline(self, model, X, y, model_info, dataset_info, 
                               is_regression=True, feature_names=None, class_names=None):
        """
        Run complete evaluation pipeline for a model
        """
        if is_regression:
            # Evaluate regression model
            metrics, y_pred = self.evaluate_regression_model(X=X, y=y, model=model, 
                                                           dataset_name=dataset_info.get('dataset_name', 'test'))
            
            # Plot regression results
            self.plot_regression_results(y, y_pred, dataset_name=dataset_info.get('dataset_name', 'test'))
            
            # Plot residuals
            self.plot_residuals(y, y_pred, dataset_name=dataset_info.get('dataset_name', 'test'))
        else:
            # Evaluate classification model
            metrics, y_pred, y_prob = self.evaluate_classification_model(X=X, y=y, model=model, 
                                                                        dataset_name=dataset_info.get('dataset_name', 'test'))
            
            # Plot confusion matrix
            self.plot_confusion_matrix(y, y_pred, class_names=class_names, 
                                      dataset_name=dataset_info.get('dataset_name', 'test'))
            
            # For binary classification, plot ROC and PR curves
            if len(np.unique(y)) <= 2 and y_prob is not None:
                self.plot_roc_curve(y, y_prob, dataset_name=dataset_info.get('dataset_name', 'test'))
                self.plot_precision_recall_curve(y, y_prob, dataset_name=dataset_info.get('dataset_name', 'test'))
        
        # Plot permutation importance if feature names are provided
        if feature_names is not None:
            importance_df = self.plot_permutation_importance(model, X, y, feature_names=feature_names, 
                                                           dataset_name=dataset_info.get('dataset_name', 'test'))
        else:
            importance_df = None
        
        # Save evaluation report
        report_path = self.save_evaluation_report(metrics, model_info, dataset_info)
        
        return metrics, importance_df, report_path 