# src/feature_engineering.py
"""
Feature engineering, transformation, and selection.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from sklearn.feature_selection import RFE, SelectKBest, f_classif, mutual_info_classif
from sklearn.inspection import permutation_importance
from imblearn.over_sampling import SMOTE
from typing import List, Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class FeatureEngineering:
    """Create and transform features."""
    
    def __init__(self):
        self.created_features = []
        self.categorical_encoders = {}
    
    def create_temporal_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal features from date columns.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new temporal features
        """
        df = df.copy()
        
        if 'Month' in df.columns:
            # Create seasonal features
            month_mapping = {
                'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
                'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
            }
            if isinstance(df['Month'].iloc[0], str):
                df['Month'] = df['Month'].map(month_mapping)
            
            # Season: 0=Winter, 1=Spring, 2=Summer, 3=Fall
            df['Season'] = df['Month'].apply(lambda x: (x - 1) // 3)
            self.created_features.append('Season')
            
            # Is holiday season (Nov-Dec)
            df['IsHolidaySeason'] = df['Month'].isin([11, 12]).astype(int)
            self.created_features.append('IsHolidaySeason')
        
        logger.info(f"Created {len(self.created_features)} temporal features")
        return df
    
    def create_aggregate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create aggregate features from existing numerical features.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with new aggregate features
        """
        df = df.copy()
        
        # Total pages visited
        if all(col in df.columns for col in ['Administrative', 'Informational', 'ProductRelated']):
            df['TotalPages'] = (df['Administrative'] + 
                               df['Informational'] + 
                               df['ProductRelated'])
            self.created_features.append('TotalPages')
            
            # Pages per page type ratios
            df['AdminPageRatio'] = df['Administrative'] / (df['TotalPages'] + 1)
            df['InfoPageRatio'] = df['Informational'] / (df['TotalPages'] + 1)
            df['ProdPageRatio'] = df['ProductRelated'] / (df['TotalPages'] + 1)
            self.created_features.extend(['AdminPageRatio', 'InfoPageRatio', 'ProdPageRatio'])
        
        # Total duration
        if all(col in df.columns for col in ['Administrative_Duration', 'Informational_Duration', 
                                              'ProductRelated_Duration']):
            df['TotalDuration'] = (df['Administrative_Duration'] + 
                                   df['Informational_Duration'] + 
                                   df['ProductRelated_Duration'])
            self.created_features.append('TotalDuration')
            
            # Average duration per page
            df['AvgDurationPerPage'] = df['TotalDuration'] / (df['TotalPages'] + 1)
            self.created_features.append('AvgDurationPerPage')
        
        # Bounce/Exit ratio
        if all(col in df.columns for col in ['BounceRate', 'ExitRate']):
            df['BounceExitRatio'] = df['BounceRate'] / (df['ExitRate'] + 0.001)
            self.created_features.append('BounceExitRatio')
        
        logger.info(f"Created {len(self.created_features)} aggregate features")
        return df
    
    def encode_categorical(self, df: pd.DataFrame, categorical_cols: List[str],
                          method: str = 'onehot', fit: bool = True) -> pd.DataFrame:
        """Encode categorical variables.
        
        Args:
            df: Input DataFrame
            categorical_cols: List of categorical column names
            method: 'onehot' or 'label'
            fit: Whether to fit the encoder (set False for test data)
            
        Returns:
            DataFrame with encoded categorical variables
        """
        df = df.copy()
        
        if method == 'onehot':
            # One-hot encoding
            df_encoded = pd.get_dummies(df[categorical_cols], drop_first=True)
            df = pd.concat([df.drop(categorical_cols, axis=1), df_encoded], axis=1)
            logger.info(f"One-hot encoded {len(categorical_cols)} categorical features")
        
        elif method == 'label':
            # Label encoding
            for col in categorical_cols:
                if col not in self.categorical_encoders and fit:
                    encoder = LabelEncoder()
                    df[col] = encoder.fit_transform(df[col].astype(str))
                    self.categorical_encoders[col] = encoder
                elif col in self.categorical_encoders:
                    df[col] = self.categorical_encoders[col].transform(df[col].astype(str))
            logger.info(f"Label encoded {len(categorical_cols)} categorical features")
        
        return df


class FeatureSelector:
    """Select optimal features using various methods."""
    
    def __init__(self, method: str = 'rfe', n_features: int = 15, cv: int = 5):
        """Initialize feature selector.
        
        Args:
            method: 'rfe', 'selectkbest', 'permutation', 'correlation'
            n_features: Number of features to select
            cv: Number of cross-validation folds
        """
        self.method = method
        self.n_features = n_features
        self.cv = cv
        self.selector = None
        self.selected_features = None
        self.feature_scores = None
    
    def select_features_rfe(self, X: pd.DataFrame, y: pd.Series,
                            estimator=None) -> List[str]:
        """Select features using Recursive Feature Elimination.
        
        Args:
            X: Features DataFrame
            y: Target Series
            estimator: Model to use for RFE (default: LogisticRegression)
            
        Returns:
            List of selected feature names
        """
        if estimator is None:
            from sklearn.linear_model import LogisticRegression
            estimator = LogisticRegression(max_iter=1000, random_state=42)
        
        self.selector = RFE(estimator, n_features_to_select=self.n_features)
        self.selector.fit(X, y)
        
        self.selected_features = X.columns[self.selector.support_].tolist()
        logger.info(f"RFE selected {len(self.selected_features)} features")
        return self.selected_features
    
    def select_features_kbest(self, X: pd.DataFrame, y: pd.Series,
                              score_func=f_classif) -> List[str]:
        """Select features using SelectKBest.
        
        Args:
            X: Features DataFrame
            y: Target Series
            score_func: Scoring function (f_classif, mutual_info_classif)
            
        Returns:
            List of selected feature names
        """
        self.selector = SelectKBest(score_func=score_func, k=self.n_features)
        self.selector.fit(X, y)
        
        # Get scores for each feature
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'score': self.selector.scores_
        }).sort_values('score', ascending=False)
        
        self.selected_features = feature_scores.head(self.n_features)['feature'].tolist()
        self.feature_scores = feature_scores
        logger.info(f"SelectKBest selected {len(self.selected_features)} features")
        return self.selected_features
    
    def select_features_permutation(self, model, X: pd.DataFrame, y: pd.Series) -> List[str]:
        """Select features using Permutation Importance.
        
        Args:
            model: Trained model
            X: Features DataFrame
            y: Target Series
            
        Returns:
            List of selected feature names
        """
        perm_importance = permutation_importance(model, X, y, n_repeats=10, 
                                                 random_state=42, n_jobs=-1)
        
        feature_scores = pd.DataFrame({
            'feature': X.columns,
            'importance': perm_importance.importances_mean
        }).sort_values('importance', ascending=False)
        
        self.selected_features = feature_scores.head(self.n_features)['feature'].tolist()
        self.feature_scores = feature_scores
        logger.info(f"Permutation importance selected {len(self.selected_features)} features")
        return self.selected_features
    
    def select_features_correlation(self, X: pd.DataFrame, y: pd.Series,
                                   threshold: float = 0.5) -> List[str]:
        """Select features based on correlation with target.
        
        Args:
            X: Features DataFrame
            y: Target Series
            threshold: Correlation threshold for target
            
        Returns:
            List of selected feature names
        """
        # Calculate correlation with target
        correlations = pd.DataFrame({
            'feature': X.columns,
            'correlation': [X[col].corr(y) for col in X.columns]
        })
        correlations['abs_correlation'] = correlations['correlation'].abs()
        correlations = correlations.sort_values('abs_correlation', ascending=False)
        
        # Select features with high correlation
        selected = correlations[correlations['abs_correlation'] >= threshold].head(self.n_features)
        self.selected_features = selected['feature'].tolist()
        self.feature_scores = correlations
        
        logger.info(f"Correlation-based selection selected {len(self.selected_features)} features")
        return self.selected_features
    
    def remove_low_variance(self, X: pd.DataFrame, threshold: float = 0.01) -> pd.DataFrame:
        """Remove features with low variance.
        
        Args:
            X: Features DataFrame
            threshold: Variance threshold
            
        Returns:
            DataFrame with low variance features removed
        """
        variances = X.var()
        low_var_features = variances[variances < threshold].index.tolist()
        
        X_filtered = X.drop(low_var_features, axis=1)
        logger.info(f"Removed {len(low_var_features)} low variance features")
        return X_filtered
    
    def remove_high_correlation(self, X: pd.DataFrame, threshold: float = 0.95) -> pd.DataFrame:
        """Remove highly correlated features.
        
        Args:
            X: Features DataFrame
            threshold: Correlation threshold
            
        Returns:
            DataFrame with highly correlated features removed
        """
        corr_matrix = X.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
        
        X_filtered = X.drop(to_drop, axis=1)
        logger.info(f"Removed {len(to_drop)} highly correlated features")
        return X_filtered


class ClassBalancer:
    """Handle class imbalance using SMOTE and class weights."""
    
    @staticmethod
    def apply_smote(X: pd.DataFrame, y: pd.Series, sampling_ratio: float = 0.5) -> Tuple:
        """Apply SMOTE to balance classes.
        
        Args:
            X: Features DataFrame
            y: Target Series
            sampling_ratio: Ratio for minority class (0-1)
            
        Returns:
            Tuple of (X_balanced, y_balanced)
        """
        smote = SMOTE(sampling_strategy=sampling_ratio, random_state=42)
        X_balanced, y_balanced = smote.fit_resample(X, y)
        
        logger.info(f"SMOTE applied. New class distribution: {y_balanced.value_counts().to_dict()}")
        return X_balanced, y_balanced
    
    @staticmethod
    def get_class_weights(y: pd.Series) -> dict:
        """Calculate class weights for imbalanced data.
        
        Args:
            y: Target Series
            
        Returns:
            Dictionary of class weights
        """
        from sklearn.utils.class_weight import compute_class_weight
        
        classes = np.unique(y)
        weights = compute_class_weight('balanced', classes=classes, y=y)
        class_weights = {classes[i]: weights[i] for i in range(len(classes))}
        
        logger.info(f"Class weights: {class_weights}")
        return class_weights
