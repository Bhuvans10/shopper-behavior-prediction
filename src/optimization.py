# src/optimization.py
"""
Hyperparameter optimization and tuning.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
import optuna
from optuna.samplers import TPESampler
from typing import Dict, Any, Tuple
import logging
import time

logger = logging.getLogger(__name__)


class HyperparameterOptimizer:
    """Optimize hyperparameters using various methods."""
    
    def __init__(self, cv: int = 5, random_state: int = 42):
        self.cv = cv
        self.random_state = random_state
        self.best_params = {}
        self.best_score = 0
    
    def optimize_with_gridsearch(self, model, param_grid: Dict[str, list],
                                X_train: pd.DataFrame, y_train: pd.Series,
                                scoring: str = 'roc_auc') -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using GridSearchCV.
        
        Args:
            model: Model to optimize
            param_grid: Parameter grid to search
            X_train: Training features
            y_train: Training target
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_params, best_score)
        """
        start_time = time.time()
        
        grid_search = GridSearchCV(
            model, param_grid, cv=self.cv, scoring=scoring,
            n_jobs=-1, verbose=1
        )
        grid_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        self.best_params = grid_search.best_params_
        self.best_score = grid_search.best_score_
        
        logger.info(f"GridSearchCV completed in {elapsed_time:.2f}s")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def optimize_with_randomsearch(self, model, param_dist: Dict[str, list],
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  n_iter: int = 20, scoring: str = 'roc_auc') -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using RandomizedSearchCV.
        
        Args:
            model: Model to optimize
            param_dist: Parameter distribution to search
            X_train: Training features
            y_train: Training target
            n_iter: Number of iterations
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_params, best_score)
        """
        start_time = time.time()
        
        random_search = RandomizedSearchCV(
            model, param_dist, n_iter=n_iter, cv=self.cv,
            scoring=scoring, n_jobs=-1, random_state=self.random_state, verbose=1
        )
        random_search.fit(X_train, y_train)
        
        elapsed_time = time.time() - start_time
        self.best_params = random_search.best_params_
        self.best_score = random_search.best_score_
        
        logger.info(f"RandomizedSearchCV completed in {elapsed_time:.2f}s")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def optimize_with_optuna(self, model_class, param_space: Dict[str, Any],
                            X_train: pd.DataFrame, y_train: pd.Series,
                            n_trials: int = 100, scoring: str = 'roc_auc') -> Tuple[Dict[str, Any], float]:
        """Optimize hyperparameters using Optuna.
        
        Args:
            model_class: Model class (not instance)
            param_space: Parameter space definition
            X_train: Training features
            y_train: Training target
            n_trials: Number of trials
            scoring: Scoring metric
            
        Returns:
            Tuple of (best_params, best_score)
        """
        def objective(trial):
            # Suggest hyperparameters
            params = {}
            for param_name, param_values in param_space.items():
                if isinstance(param_values[0], int):
                    params[param_name] = trial.suggest_int(param_name, param_values[0], param_values[1])
                else:
                    params[param_name] = trial.suggest_float(param_name, param_values[0], param_values[1])
            
            # Create and train model
            model = model_class(**params)
            scores = cross_val_score(model, X_train, y_train, cv=self.cv, scoring=scoring)
            
            return scores.mean()
        
        start_time = time.time()
        
        sampler = TPESampler(seed=self.random_state)
        study = optuna.create_study(direction='maximize', sampler=sampler)
        study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
        
        elapsed_time = time.time() - start_time
        self.best_params = study.best_params
        self.best_score = study.best_value
        
        logger.info(f"Optuna optimization completed in {elapsed_time:.2f}s")
        logger.info(f"Best params: {self.best_params}")
        logger.info(f"Best score: {self.best_score:.4f}")
        
        return self.best_params, self.best_score
    
    def learning_curves(self, model, X_train: pd.DataFrame, y_train: pd.Series,
                       train_sizes: np.ndarray = None, cv: int = 5) -> Tuple[np.ndarray, np.ndarray]:
        """Generate learning curves.
        
        Args:
            model: Model instance
            X_train: Training features
            y_train: Training target
            train_sizes: Training set sizes to use
            cv: Number of cross-validation folds
            
        Returns:
            Tuple of (train_scores, val_scores)
        """
        from sklearn.model_selection import learning_curve
        
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)
        
        train_sizes_abs, train_scores, val_scores = learning_curve(
            model, X_train, y_train, cv=cv, train_sizes=train_sizes,
            scoring='roc_auc', n_jobs=-1
        )
        
        logger.info(f"Learning curves generated for {len(train_sizes_abs)} training sizes")
        
        return train_sizes_abs, train_scores, val_scores
