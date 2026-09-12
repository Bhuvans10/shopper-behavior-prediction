# src/models.py
"""
Model definitions and utilities.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from typing import Dict, Any, List
import logging

logger = logging.getLogger(__name__)


def get_baseline_model(**params) -> LogisticRegression:
    """Get baseline logistic regression model.
    
    Args:
        **params: Model hyperparameters
        
    Returns:
        Logistic Regression model
    """
    default_params = {
        'C': 1.0,
        'max_iter': 1000,
        'solver': 'lbfgs',
        'random_state': 42,
        'class_weight': 'balanced'
    }
    default_params.update(params)
    
    model = LogisticRegression(**default_params)
    logger.info(f"Baseline model created: LogisticRegression")
    return model


def get_random_forest(**params) -> RandomForestClassifier:
    """Get Random Forest model.
    
    Args:
        **params: Model hyperparameters
        
    Returns:
        Random Forest Classifier
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 5,
        'min_samples_leaf': 2,
        'max_features': 'sqrt',
        'random_state': 42,
        'n_jobs': -1,
        'class_weight': 'balanced'
    }
    default_params.update(params)
    
    model = RandomForestClassifier(**default_params)
    logger.info(f"Random Forest model created")
    return model


def get_xgboost(**params) -> XGBClassifier:
    """Get XGBoost model.
    
    Args:
        **params: Model hyperparameters
        
    Returns:
        XGBoost Classifier
    """
    default_params = {
        'n_estimators': 100,
        'max_depth': 5,
        'learning_rate': 0.1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'random_state': 42,
        'scale_pos_weight': 1
    }
    default_params.update(params)
    
    model = XGBClassifier(**default_params)
    logger.info(f"XGBoost model created")
    return model


def get_lightgbm(**params) -> LGBMClassifier:
    """Get LightGBM model.
    
    Args:
        **params: Model hyperparameters
        
    Returns:
        LightGBM Classifier
    """
    default_params = {
        'n_estimators': 100,
        'num_leaves': 31,
        'max_depth': 5,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'objective': 'binary',
        'metric': 'auc',
        'random_state': 42,
        'is_unbalance': True,
        'verbose': -1
    }
    default_params.update(params)
    
    model = LGBMClassifier(**default_params)
    logger.info(f"LightGBM model created")
    return model


def get_voting_ensemble(models: List, weights: List[int] = None, 
                       voting: str = 'soft') -> VotingClassifier:
    """Create Voting Ensemble.
    
    Args:
        models: List of tuples (name, model)
        weights: Weights for each model
        voting: 'soft' or 'hard'
        
    Returns:
        Voting Classifier
    """
    ensemble = VotingClassifier(
        estimators=models,
        weights=weights,
        voting=voting
    )
    logger.info(f"Voting Ensemble created with {len(models)} models")
    return ensemble


def get_stacking_ensemble(base_models: List, final_estimator=None) -> StackingClassifier:
    """Create Stacking Ensemble.
    
    Args:
        base_models: List of tuples (name, model) for base learners
        final_estimator: Meta-learner model
        
    Returns:
        Stacking Classifier
    """
    if final_estimator is None:
        final_estimator = LogisticRegression(max_iter=1000, random_state=42)
    
    ensemble = StackingClassifier(
        estimators=base_models,
        final_estimator=final_estimator,
        cv=5
    )
    logger.info(f"Stacking Ensemble created with {len(base_models)} base models")
    return ensemble


def get_all_models(config: Dict[str, Any] = None) -> Dict[str, Any]:
    """Get all model configurations.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Dictionary of models
    """
    models = {
        'baseline': get_baseline_model(),
        'random_forest': get_random_forest(),
        'xgboost': get_xgboost(),
        'lightgbm': get_lightgbm(),
    }
    
    # Create ensemble
    base_models = [
        ('rf', models['random_forest']),
        ('xgb', models['xgboost']),
        ('lgb', models['lightgbm'])
    ]
    models['voting'] = get_voting_ensemble(base_models, weights=[1, 2, 2])
    
    logger.info(f"Created {len(models)} models")
    return models
