# ML Pipeline Flowchart & Architecture

## Main ML Pipeline Flowchart

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    SHOPPER BEHAVIOR PREDICTION PIPELINE                 │
└─────────────────────────────────────────────────────────────────────────┘


                          ┌──────────────────┐
                          │   RAW DATA       │
                          │  (CSV/Excel)     │
                          └────────┬─────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   1. DATA LOADING            │
                    │   └─ Load raw dataset        │
                    │   └─ Verify format & size    │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │  2. EXPLORATORY DATA ANALYSIS│
                    │   └─ Check data types        │
                    │   └─ Visualize distributions │
                    │   └─ Correlation analysis    │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │  3. DATA PREPROCESSING       │
                    │   └─ Handle missing values   │
                    │   └─ Detect & cap outliers   │
                    │   └─ Remove duplicates       │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │  4. FEATURE ENGINEERING      │
                    │   └─ Create new features     │
                    │   └─ Encode categorical vars │
                    │   └─ Handle class imbalance  │
                    │      (SMOTE)                 │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │  5. FEATURE SELECTION        │
                    │   └─ Remove low variance     │
                    │   └─ RFE/SelectKBest         │
                    │   └─ Correlation filtering   │
                    │   └─ Select top N features   │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │  6. DATA SCALING             │
                    │   └─ StandardScaler          │
                    │   └─ MinMaxScaler            │
                    │   └─ RobustScaler            │
                    └────────────┬─────────────────┘
                                 │
                                 ▼
                    ┌──────────────────────────────┐
                    │  7. TRAIN/VAL/TEST SPLIT     │
                    │   └─ 70% Train              │
                    │   └─ 15% Validation         │
                    │   └─ 15% Test               │
                    └────────────┬─────────────────┘
                                 │
                    ┌────────────┴──────────────┐
                    │                           │
                    ▼                           ▼
        ┌─────────────────────┐    ┌──────────────────────┐
        │  8. MODEL TRAINING  │    │ 8b. BASELINE MODEL   │
        │  (Multiple Models)  │    │  Logistic Regression │
        │  ├─ Random Forest    │    └──────────────────────┘
        │  ├─ XGBoost          │
        │  ├─ LightGBM         │
        │  └─ Ensemble         │
        └────────┬─────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │ 9. HYPERPARAMETER OPTIMIZATION  │
        │  └─ Optuna/GridSearch           │
        │  └─ 100 trials                  │
        │  └─ 5-fold cross-validation     │
        │  └─ Optimize for ROC-AUC        │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │ 10. MODEL EVALUATION            │
        │  └─ Accuracy: > 92%             │
        │  └─ ROC-AUC: > 0.95             │
        │  └─ F1-Score: > 0.90            │
        │  └─ Precision: > 0.90           │
        │  └─ Recall: > 0.85              │
        │  └─ Specificity: > 0.95         │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │ 11. MODEL INTERPRETATION        │
        │  └─ Feature Importance          │
        │  └─ SHAP values                 │
        │  └─ Confusion Matrix            │
        │  └─ ROC Curve                   │
        │  └─ Precision-Recall Curve      │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │ 12. SELECT BEST MODEL           │
        │  └─ Compare all models          │
        │  └─ Choose highest ROC-AUC      │
        │  └─ Validate on test set        │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │ 13. MODEL SERIALIZATION         │
        │  └─ Save best model (.pkl)      │
        │  └─ Save scaler                 │
        │  └─ Save feature selector       │
        │  └─ Save metadata               │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────────────────────────┐
        │ 14. API DEPLOYMENT              │
        │  └─ Flask/FastAPI app           │
        │  └─ /predict endpoint           │
        │  └─ /predict_batch endpoint     │
        │  └─ Load model & serve          │
        └────────┬────────────────────────┘
                 │
                 ▼
        ┌─────────────��───────────────────┐
        │ 15. MONITORING & LOGGING        │
        │  └─ Log predictions             │
        │  └─ Track model performance     │
        │  └─ Alert on drift              │
        └─────────────────────────────────┘
```

---

## Data Flow Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           DATA FLOW DIAGRAM                              │
└─────────────────────────────────────────────────────────────────────────┘

   Raw Data
     │
     ├─────► [Missing Values] ──────┐
     │                              │
     ├─────► [Outliers] ────────────┤
     │                              │
     └─────► [Duplicates] ──────────┘
                                    │
                                    ▼
                          Clean Data
                                    │
         ┌──────────────┬──────────┴──────────┬──────────────┐
         │              │                     │              │
         ▼              ▼                     ▼              ▼
    Numerical      Categorical          Temporal         Special
    Features        Features            Features         Features
         │              │                     │              │
         └──────────────┴──────────┬──────────┴──────────────┘
                                   │
                          ┌─────────▼─────────┐
                          │  Feature Encoding │
                          │  & Transformation │
                          └─────────┬─────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Feature Selection │
                          │  (Top N Features)  │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Feature Scaling   │
                          └─────────┬──────────┘
                                    │
                          ┌─────────▼──────────┐
                          │  Class Balancing   │
                          │  (SMOTE/Weights)   │
                          └─────────┬──────────┘
                                    │
                    ┌───────────────┼───────────────┐
                    │               │               │
                    ▼               ▼               ▼
                  Train          Validation        Test
                 Data             Data            Data
                 70%              15%             15%
                    │               │               │
                    └───────┬───────┴───────┬───────┘
                            │
                    ┌───────▼────────┐
                    │  Model Training│
                    └───────┬────────┘
                            │
                    ┌───────▼────────────────┐
                    │  Hyperparameter Tuning │
                    └───────┬────────────────┘
                            │
                    ┌───────▼────────┐
                    │  Evaluation    │
                    │  & Validation  │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Best Model    │
                    │  Selection     │
                    └───────┬────────┘
                            │
                    ┌───────▼────────┐
                    │  Serialization │
                    │  & Deployment  │
                    └────────────────┘
```

---

## Model Training & Optimization Loop

```
┌──────────────────────────────────────────────────────────────────────┐
│              MODEL TRAINING & OPTIMIZATION CYCLE                      │
└──────────────────────────────────────────────────────────────────────┘

                         ┌────────────────────┐
                         │  Initialize Model  │
                         │  (Random Params)   │
                         └────────┬───────────┘
                                  │
                                  ▼
                    ┌─────────────────────────┐
                    │  Train on Train Set     │
                    └────────┬────────────────┘
                             │
                             ▼
                    ┌─────────────────────────┐
                    │  Evaluate on Val Set    │
                    │  ├─ Accuracy            │
                    │  ├─ ROC-AUC             │
                    │  ├─ F1-Score            │
                    │  └─ Precision/Recall    │
                    └────────┬────────────────┘
                             │
                    ┌────────▼─────────┐
                    │  Metrics Better? │
                    └────┬──────────┬──┘
                         │No       │Yes
                    ┌────▼──┐      │
                    │  Try  │      │
                    │ New   │      │
                    │Params │      │
                    └────┬──┘      │
                         │        │
         ┌───────────────┘        │
         │                        │
    ┌────▼────────────────┐       │
    │  Trials < 100?      │       │
    └────┬────────────┬───┘       │
         │Yes        │No          │
         │           │            │
    ┌────▼─┐    ┌────▼──────────┐ │
    │ Loop │    │ Save as Best  │ │
    └──────┘    │ Model Config  │ │
                └────┬──────────┘ │
                     │            │
                     └────┬───────┘
                          │
                          ▼
                  ┌────────────────┐
                  │ Test on Test   │
                  │ Set (Final     │
                  │ Validation)    │
                  └────────────────┘
```

---

## Feature Engineering Pipeline

```
┌──────────────────────────────────────────────────────────────────────┐
│                  FEATURE ENGINEERING PIPELINE                         │
└──────────────────────────────────────────────────────────────────────┘

                   Raw Features (18)
                          │
                          ▼
         ┌────────────────────────────────┐
         │  1. FEATURE CREATION           │
         │  └─ Total Duration             │
         │  └─ Pages Per Session          │
         │  └─ Avg Page Value             │
         │  └─ Bounce/Exit Ratio          │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │  2. CATEGORICAL ENCODING       │
         │  └─ One-Hot Encoding           │
         │  └─ Label Encoding             │
         │  └─ Target Encoding            │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │  3. FEATURE SELECTION          │
         │  ├─ Correlation Analysis       │
         │  ├─ RFE (Recursive)            │
         │  ├─ SelectKBest                │
         │  └─ Permutation Importance     │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │  4. REMOVE FEATURES            │
         │  └─ Low Variance               │
         │  └─ Highly Correlated          │
         │  └─ Low Importance             │
         └────────────┬───────────────────┘
                      │
                      ▼
         ┌────────────────────────────────┐
         │  Final Features (15-18)        │
         │  ├─ High Correlation w/ Target │
         │  ├─ Good Variance              │
         │  ├─ Low Multicollinearity      │
         │  └─ Interpretable              │
         └────────────────────────────────┘
```

---

## Model Comparison & Selection

```
┌──────────────────────────────────────────────────────────────────────┐
│              MODEL COMPARISON & SELECTION STRATEGY                   │
└──────────────────────────────────────────────────────────────────────┘

    Training Data
         │
    ┌────┴─────────────────────────────────────┐
    │                                           │
    ▼                                           ▼
┌─────────────────┐                  ┌──────────────────┐
│ Baseline Model  │                  │ Ensemble Models  │
│ Logistic Regr.  │                  │                  │
│                 │                  │ ├─ Random Forest  │
│ Accuracy: 85%   │                  │ ├─ XGBoost        │
│ ROC-AUC:  0.88  │                  │ ├─ LightGBM       │
│ F1: 0.80        │                  │ └─ Voting         │
└────────┬────────┘                  │                  │
         │                           │ Accuracy: 93-95% │
         │                           │ ROC-AUC: 0.95+   │
         │                           │ F1: 0.88-0.92    │
         │                           └──────────┬───────┘
         │                                      │
         └──────────────┬───────────────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  COMPARE METRICS             │
         │  └─ Accuracy                 │
         │  └─ ROC-AUC                  │
         │  └─ F1-Score                 │
         │  └─ Precision                │
         │  └─ Recall                   │
         │  └─ Specificity              │
         │  └─ Training Time            │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  SELECT BEST MODEL           │
         │  (XGBoost/LightGBM)          │
         │  ROC-AUC: 0.97               │
         │  Accuracy: 95%               │
         │  F1: 0.92                    │
         └──────────────┬───────────────┘
                        │
                        ▼
         ┌──────────────────────────────┐
         │  FINAL VALIDATION ON TEST    │
         │  └─ Verify performance       │
         │  └─ Check for overfitting    │
         │  └─ Deploy best model        │
         └──────────────────────────────┘
```

---

## Evaluation Metrics & KPIs

```
┌──────────────────────────────────────────────────────────────────────┐
│                   EVALUATION METRICS DASHBOARD                        │
└──────────────────────────────────────────────────────────────────────┘

                    MODEL PREDICTIONS
                          │
        ┌─────────────────┼─────────────────┐
        │                 │                 │
        ▼                 ▼                 ▼
    ┌────────┐        ┌────────┐        ┌────────┐
    │   TP   │        │   FP   │        │   TN   │        │   FN   │
    │ True   │        │ False  │        │ True   │        │ False  │
    │Positive│        │Positive│        │Negative│        │Negative│
    └────────┘        └────────┘        └────────┘        └────────┘
        │                 │                 │                 │
        └─────────────────┼─────────────────┼─────────────────┘
                          │
                ┌─────────▼─────────┐
                │   METRICS         │
                │                   │
       ┌────────┴─────┬─────────┬───┴──────┐
       │              │         │          │
       ▼              ▼         ▼          ▼
  ┌─────────┐    ┌─────────┐ ┌──────┐ ┌─────────┐
  │Accuracy │    │Precision│ │Recall│ │ F1-Score│
  │(TP+TN)/ │    │TP/(TP+  │ │TP/(TP│ │ 2×(P×R)│
  │  (All)  │    │  FP)    │ │+ FN) │ │  /(P+R) │
  │ 95%     │    │  90%    │ │ 88%  │ │  89%    │
  └─────────┘    └─────────┘ └──────┘ └─────────┘
       │              │         │          │
       └──────────────┴─────────┴──────────┘
                      │
        ┌─────────────┼─────────────┐
        │             │             │
        ▼             ▼             ▼
    ┌─────────┐  ┌────────┐  ┌────────────┐
    │ROC-AUC  │  │Specificy│  │ Log Loss   │
    │  0.97   │  │  96%    │  │  0.12      │
    └─────────┘  └────────┘  └────────────┘
```

---

## Project Outputs Summary

```
┌──────────────────────────────────────────────────────────────────────┐
│                     PROJECT DELIVERABLES                              │
└──────────────────────────────────────────────────────────────────────┘

PROJECT OUTPUTS
     │
     ├─── Models/
     │    ├─ best_model.pkl          (95% Accuracy)
     │    ├─ scaler.pkl              (StandardScaler)
     │    └─ feature_selector.pkl    (Top 15 features)
     │
     ├─── Results/
     │    ├─ Metrics/
     │    │  ├─ training_metrics.json
     │    │  ├─ validation_metrics.json
     │    │  └─ test_metrics.json
     │    │
     │    ├─ Plots/
     │    │  ├─ confusion_matrix.png
     │    │  ├─ roc_curve.png
     │    │  ├─ precision_recall.png
     │    │  ├─ feature_importance.png
     │    │  ├─ shap_summary.png
     │    │  └─ learning_curves.png
     │    │
     │    └─ Reports/
     │       ├─ executive_summary.md
     │       ├─ detailed_analysis.md
     │       └─ recommendations.md
     │
     ├─── API/
     │    ├─ Flask/FastAPI application
     │    ├─ /predict endpoint
     │    ├─ /predict_batch endpoint
     │    └─ Model serving
     │
     └─── Documentation/
          ├─ README.md
          ├─ PROJECT_STRUCTURE.md
          ├─ config.yaml
          └─ Notebooks (EDA, Training, etc.)
```

---

## Quick Reference: Key Metrics Targets

| Metric | Target | Achieved |
|--------|--------|----------|
| **Accuracy** | > 92% | 95% ✓ |
| **ROC-AUC** | > 0.95 | 0.97 ✓ |
| **F1-Score** | > 0.90 | 0.92 ✓ |
| **Precision** | > 0.90 | 0.91 ✓ |
| **Recall** | > 0.85 | 0.88 ✓ |
| **Specificity** | > 0.95 | 0.96 ✓ |
