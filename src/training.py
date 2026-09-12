# src/training.py
"""
Model training pipeline.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, Any, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class ModelTrainer:
    """Train and evaluate models."""
    
    def __init__(self):
        self.trained_models = {}
        self.training_history = {}
    
    def fit_model(self, name: str, model, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        """Fit a model on training data.
        
        Args:
            name: Model name
            model: Model instance
            X_train: Training features
            y_train: Training target
        """
        start_time = time.time()
        model.fit(X_train, y_train)
        training_time = time.time() - start_time
        
        self.trained_models[name] = model
        self.training_history[name] = {'training_time': training_time}
        
        logger.info(f"Model '{name}' trained in {training_time:.2f}s")
    
    def evaluate_on_validation(self, name: str, model, X_val: pd.DataFrame, 
                               y_val: pd.Series) -> Dict[str, float]:
        """Evaluate model on validation set.
        
        Args:
            name: Model name
            model: Model instance
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary of metrics
        """
        y_pred = model.predict(X_val)
        y_proba = model.predict_proba(X_val)[:, 1] if hasattr(model, 'predict_proba') else None
        
        metrics = {
            'accuracy': accuracy_score(y_val, y_pred),
            'precision': precision_score(y_val, y_pred),
            'recall': recall_score(y_val, y_pred),
            'f1': f1_score(y_val, y_pred),
        }
        
        if y_proba is not None:
            metrics['roc_auc'] = roc_auc_score(y_val, y_proba)
        
        if name in self.training_history:
            self.training_history[name].update(metrics)
        
        logger.info(f"Model '{name}' validation metrics: {metrics}")
        return metrics
    
    def cross_validate_model(self, name: str, model, X: pd.DataFrame, y: pd.Series,
                            cv: int = 5) -> Dict[str, np.ndarray]:
        """Perform cross-validation on model.
        
        Args:
            name: Model name
            model: Model instance
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary of cross-validation scores
        """
        scoring = ['accuracy', 'precision', 'recall', 'f1', 'roc_auc']
        cv_results = cross_validate(model, X, y, cv=cv, scoring=scoring)
        
        logger.info(f"Model '{name}' cross-validation completed")
        return cv_results
    
    def train_multiple_models(self, models: Dict[str, Any], X_train: pd.DataFrame,
                             y_train: pd.Series, X_val: pd.DataFrame, 
                             y_val: pd.Series) -> Dict[str, Dict[str, float]]:
        """Train multiple models and evaluate on validation set.
        
        Args:
            models: Dictionary of model name to model instance
            X_train: Training features
            y_train: Training target
            X_val: Validation features
            y_val: Validation target
            
        Returns:
            Dictionary of model names to metrics
        """
        results = {}
        
        for name, model in models.items():
            logger.info(f"Training model: {name}")
            self.fit_model(name, model, X_train, y_train)
            metrics = self.evaluate_on_validation(name, model, X_val, y_val)
            results[name] = metrics
        
        logger.info(f"Trained {len(models)} models")
        return results
    
    def get_best_model(self, metric: str = 'roc_auc') -> Tuple[str, Any, float]:
        """Get best performing model based on metric.
        
        Args:
            metric: Metric to use for ranking
            
        Returns:
            Tuple of (model_name, model, metric_value)
        """
        best_name = None
        best_model = None
        best_value = -1
        
        for name, model in self.trained_models.items():
            if name in self.training_history and metric in self.training_history[name]:
                value = self.training_history[name][metric]
                if value > best_value:
                    best_value = value
                    best_name = name
                    best_model = model
        
        logger.info(f"Best model: {best_name} with {metric}={best_value:.4f}")
        return best_name, best_model, best_value
    
    def get_training_summary(self) -> pd.DataFrame:
        """Get summary of all trained models.
        
        Returns:
            DataFrame with model metrics
        """
        summary_data = []
        
        for name, metrics in self.training_history.items():
            row = {'model': name}
            row.update(metrics)
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        logger.info("Training summary:")
        logger.info(summary_df.to_string())
        return summary_df
