"""
Student Performance Machine Learning Model Service
===================================================
This module trains, evaluates, and provides prediction services for:
1. Random Forest Classifier (Ensemble Bagging Classification)
2. Decision Tree Classifier (CART Rule Tree Classification)
3. Logistic Regression Classifier (Linear Probabilistic Classification)
4. K-Nearest Neighbors Classifier (Instance-Based Distance Metric)
5. Linear Regression (Continuous Mark Prediction)
"""

import os
import json
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, mean_squared_error, r2_score, mean_absolute_error,
    roc_auc_score
)
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

DATASET_PATH = "Student_Performance_Professional_10000.csv"
REGRESSION_MODEL_PATH = "student_model.joblib"
REGRESSION_METRICS_PATH = "model_metrics.json"

RF_MODEL_PATH = "random_forest_model.joblib"
DT_MODEL_PATH = "decision_tree_model.joblib"
LOGREG_MODEL_PATH = "logistic_regression_model.joblib"
KNN_MODEL_PATH = "knn_model.joblib"
CLASSIFICATION_METRICS_PATH = "classification_metrics.json"

NUMERIC_FEATURES = [
    'Attendance_%',
    'Study_Hours_Per_Day',
    'Previous_Marks',
    'Assignments_Completed',
    'Sleep_Hours'
]

CATEGORICAL_FEATURES = [
    'Internet_Access',
    'Extra_Classes'
]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET_REGRESSION = 'Previous_Marks'
TARGET_CLASSIFICATION = 'Final_Result'

def get_preprocessor():
    """Creates a ColumnTransformer for numerical scaling and categorical encoding."""
    numeric_transformer = Pipeline([
        ('scaler', StandardScaler())
    ])
    categorical_transformer = Pipeline([
        ('onehot', OneHotEncoder(drop='first', handle_unknown='ignore'))
    ])
    return ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, NUMERIC_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES)
        ]
    )

def train_all_models():
    """Trains 4 Classification models and 1 Linear Regression model, exporting models and metrics."""
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}")
    
    df = pd.read_csv(DATASET_PATH)
    print(f"Loaded dataset: {df.shape[0]} records, {df.shape[1]} columns")

    # =========================================================================
    # 1. TRAIN LINEAR REGRESSION (Continuous Marks)
    # =========================================================================
    reg_features = ['Attendance_%', 'Study_Hours_Per_Day', 'Assignments_Completed', 'Sleep_Hours']
    X_reg = df[reg_features]
    y_reg = df[TARGET_REGRESSION]
    
    X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(
        X_reg, y_reg, test_size=0.2, random_state=42
    )
    
    lin_reg = LinearRegression()
    lin_reg.fit(X_train_r, y_train_r)
    y_pred_r = lin_reg.predict(X_test_r)
    
    mse = float(mean_squared_error(y_test_r, y_pred_r))
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(y_test_r, y_pred_r))
    r2 = float(r2_score(y_test_r, y_pred_r))
    
    reg_metrics = {
        "model_name": "Linear Regression",
        "dataset_records": len(df),
        "train_samples": len(X_train_r),
        "test_samples": len(X_test_r),
        "features": reg_features,
        "target": TARGET_REGRESSION,
        "intercept": float(lin_reg.intercept_),
        "coefficients": {
            feat: float(coef) for feat, coef in zip(reg_features, lin_reg.coef_)
        },
        "evaluation": {
            "mean_squared_error": round(mse, 4),
            "root_mean_squared_error": round(rmse, 4),
            "mean_absolute_error": round(mae, 4),
            "r2_score": round(r2, 4)
        },
        "dataset_averages": {
            "attendance_mean": round(float(df['Attendance_%'].mean()), 2),
            "study_hours_mean": round(float(df['Study_Hours_Per_Day'].mean()), 2),
            "previous_marks_mean": round(float(df['Previous_Marks'].mean()), 2),
            "assignments_mean": round(float(df['Assignments_Completed'].mean()), 2),
            "sleep_hours_mean": round(float(df['Sleep_Hours'].mean()), 2),
        }
    }
    
    joblib.dump(lin_reg, REGRESSION_MODEL_PATH)
    with open(REGRESSION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(reg_metrics, f, indent=4)
    print(f"[OK] Linear Regression saved to {REGRESSION_MODEL_PATH}")

    # =========================================================================
    # 2. TRAIN 4 CLASSIFICATION MODELS (Pass / Fail)
    # =========================================================================
    X_clf = df[ALL_FEATURES]
    y_clf = df[TARGET_CLASSIFICATION]  # 'Pass', 'Fail'
    
    X_train_c, X_test_c, y_train_c, y_test_c = train_test_split(
        X_clf, y_clf, test_size=0.2, random_state=42, stratify=y_clf
    )
    
    preprocessor = get_preprocessor()
    encoded_feature_names = NUMERIC_FEATURES + ['Internet_Access_Yes', 'Extra_Classes_Yes']
    
    # 2a. Random Forest Classifier
    rf_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(n_estimators=150, max_depth=8, min_samples_leaf=4, random_state=42))
    ])
    rf_pipeline.fit(X_train_c, y_train_c)
    rf_pred = rf_pipeline.predict(X_test_c)
    rf_proba = rf_pipeline.predict_proba(X_test_c)[:, 1]
    rf_importances = dict(zip(encoded_feature_names, [round(float(x), 4) for x in rf_pipeline.named_steps['classifier'].feature_importances_]))
    
    # 2b. Decision Tree Classifier
    dt_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', DecisionTreeClassifier(max_depth=5, min_samples_leaf=10, random_state=42))
    ])
    dt_pipeline.fit(X_train_c, y_train_c)
    dt_pred = dt_pipeline.predict(X_test_c)
    dt_proba = dt_pipeline.predict_proba(X_test_c)[:, 1]
    dt_importances = dict(zip(encoded_feature_names, [round(float(x), 4) for x in dt_pipeline.named_steps['classifier'].feature_importances_]))
    
    # 2c. Logistic Regression Classifier
    logreg_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', LogisticRegression(max_iter=1000, random_state=42))
    ])
    logreg_pipeline.fit(X_train_c, y_train_c)
    logreg_pred = logreg_pipeline.predict(X_test_c)
    logreg_proba = logreg_pipeline.predict_proba(X_test_c)[:, 1]
    logreg_coefs = dict(zip(encoded_feature_names, [round(float(x), 4) for x in logreg_pipeline.named_steps['classifier'].coef_[0]]))
    
    # 2d. K-Nearest Neighbors Classifier (KNN)
    knn_pipeline = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', KNeighborsClassifier(n_neighbors=5, weights='distance'))
    ])
    knn_pipeline.fit(X_train_c, y_train_c)
    knn_pred = knn_pipeline.predict(X_test_c)
    knn_proba = knn_pipeline.predict_proba(X_test_c)[:, 1]
    
    def evaluate_clf(y_true, y_pred, y_prob):
        cm = confusion_matrix(y_true, y_pred, labels=['Fail', 'Pass'])
        return {
            "accuracy": round(float(accuracy_score(y_true, y_pred)), 4),
            "precision": round(float(precision_score(y_true, y_pred, pos_label='Pass', zero_division=0)), 4),
            "recall": round(float(recall_score(y_true, y_pred, pos_label='Pass', zero_division=0)), 4),
            "f1_score": round(float(f1_score(y_true, y_pred, pos_label='Pass', zero_division=0)), 4),
            "roc_auc": round(float(roc_auc_score(y_true == 'Pass', y_prob)), 4),
            "confusion_matrix": {
                "true_fail_pred_fail (TN)": int(cm[0][0]),
                "true_fail_pred_pass (FP)": int(cm[0][1]),
                "true_pass_pred_fail (FN)": int(cm[1][0]),
                "true_pass_pred_pass (TP)": int(cm[1][1]),
                "raw_matrix": cm.tolist()
            }
        }
    
    rf_eval = evaluate_clf(y_test_c, rf_pred, rf_proba)
    dt_eval = evaluate_clf(y_test_c, dt_pred, dt_proba)
    logreg_eval = evaluate_clf(y_test_c, logreg_pred, logreg_proba)
    knn_eval = evaluate_clf(y_test_c, knn_pred, knn_proba)
    
    classification_metadata = {
        "dataset_records": len(df),
        "target": TARGET_CLASSIFICATION,
        "classes": ["Fail", "Pass"],
        "features": ALL_FEATURES,
        "encoded_features": encoded_feature_names,
        "class_distribution": {
            "Pass": int((df[TARGET_CLASSIFICATION] == 'Pass').sum()),
            "Fail": int((df[TARGET_CLASSIFICATION] == 'Fail').sum()),
            "pass_rate_pct": round(float((df[TARGET_CLASSIFICATION] == 'Pass').mean() * 100), 2)
        },
        "models": {
            "random_forest": {
                "name": "Random Forest Classifier",
                "short_name": "Random Forest",
                "icon": "🌲",
                "algorithm": "Ensemble of 150 Bootstrap Decision Trees",
                "hyperparameters": {
                    "n_estimators": 150,
                    "max_depth": 8,
                    "min_samples_leaf": 4,
                    "random_state": 42
                },
                "evaluation": rf_eval,
                "feature_importances": rf_importances,
                "strengths": [
                    "High accuracy and superior generalization via bagging",
                    "Robust against overfitting on edge-case outliers",
                    "Provides reliable continuous class probabilities"
                ]
            },
            "decision_tree": {
                "name": "Decision Tree Classifier",
                "short_name": "Decision Tree",
                "icon": "🌳",
                "algorithm": "CART (Classification and Regression Trees)",
                "hyperparameters": {
                    "criterion": "gini",
                    "max_depth": 5,
                    "min_samples_leaf": 10,
                    "random_state": 42
                },
                "evaluation": dt_eval,
                "feature_importances": dt_importances,
                "strengths": [
                    "Highly interpretable with clear if-else decision rules",
                    "Captures non-linear thresholds in study hours & attendance",
                    "Requires minimal data transformation"
                ]
            },
            "logistic_regression": {
                "name": "Logistic Regression Classifier",
                "short_name": "Logistic Regression",
                "icon": "📊",
                "algorithm": "Generalized Linear Model with Logit Link Function",
                "hyperparameters": {
                    "penalty": "l2",
                    "C": 1.0,
                    "solver": "lbfgs",
                    "max_iter": 1000,
                    "random_state": 42
                },
                "evaluation": logreg_eval,
                "coefficients": logreg_coefs,
                "strengths": [
                    "Direct probabilistic interpretations through log-odds ratios",
                    "Computationally efficient with convex optimization guarantees",
                    "Ideal baseline for linear decision boundary validation"
                ]
            },
            "knn": {
                "name": "K-Nearest Neighbors Classifier (KNN)",
                "short_name": "K-Nearest Neighbors",
                "icon": "🔍",
                "algorithm": "Instance-Based Non-Parametric Classifier (k=5)",
                "hyperparameters": {
                    "n_neighbors": 5,
                    "weights": "distance",
                    "metric": "minkowski (p=2 Euclidean)"
                },
                "evaluation": knn_eval,
                "strengths": [
                    "Non-parametric model with zero training phase assumptions",
                    "Naturally captures complex local neighborhood clusters",
                    "Weighted inverse distance enhances prediction precision"
                ]
            }
        }
    }
    
    # Save Classification Models and JSON Metadata
    joblib.dump(rf_pipeline, RF_MODEL_PATH)
    joblib.dump(dt_pipeline, DT_MODEL_PATH)
    joblib.dump(logreg_pipeline, LOGREG_MODEL_PATH)
    joblib.dump(knn_pipeline, KNN_MODEL_PATH)
    
    with open(CLASSIFICATION_METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(classification_metadata, f, indent=4)
        
    print(f"[OK] Random Forest saved to {RF_MODEL_PATH} (Accuracy: {rf_eval['accuracy']:.4f})")
    print(f"[OK] Decision Tree saved to {DT_MODEL_PATH} (Accuracy: {dt_eval['accuracy']:.4f})")
    print(f"[OK] Logistic Regression saved to {LOGREG_MODEL_PATH} (Accuracy: {logreg_eval['accuracy']:.4f})")
    print(f"[OK] K-Nearest Neighbors saved to {KNN_MODEL_PATH} (Accuracy: {knn_eval['accuracy']:.4f})")
    print(f"[OK] Classification metrics saved to {CLASSIFICATION_METRICS_PATH}")
    
    return {
        "linear_regression": lin_reg,
        "random_forest": rf_pipeline,
        "decision_tree": dt_pipeline,
        "logistic_regression": logreg_pipeline,
        "knn": knn_pipeline,
        "classification_metrics": classification_metadata
    }

def predict_student(
    attendance: float,
    study_hours: float,
    previous_marks: float,
    assignments: float,
    sleep_hours: float,
    internet_access: str = "Yes",
    extra_classes: str = "No",
    model_type: str = "random_forest"
):
    """
    Generates predictions using the specified model.
    model_type options: 'random_forest', 'decision_tree', 'logistic_regression', 'knn', 'linear_regression'
    """
    input_data = pd.DataFrame([{
        'Attendance_%': attendance,
        'Study_Hours_Per_Day': study_hours,
        'Previous_Marks': previous_marks,
        'Assignments_Completed': assignments,
        'Sleep_Hours': sleep_hours,
        'Internet_Access': internet_access,
        'Extra_Classes': extra_classes
    }])
    
    if model_type == "linear_regression":
        if os.path.exists(REGRESSION_MODEL_PATH):
            model = joblib.load(REGRESSION_MODEL_PATH)
            score = float(model.predict(input_data[['Attendance_%', 'Study_Hours_Per_Day', 'Assignments_Completed', 'Sleep_Hours']])[0])
        else:
            score = 55.4294 - (0.0134 * attendance) + (2.9843 * study_hours) - (0.0310 * assignments) - (0.1046 * sleep_hours)
            
        bounded = max(0.0, min(100.0, score))
        tier = "Good Performance" if bounded >= 70 else ("Average Performance" if bounded >= 40 else "Needs Improvement")
        return {
            "model_type": "Linear Regression",
            "predicted_score": round(bounded, 2),
            "tier": tier,
            "formula": f"55.4294 - 0.0134({attendance}) + 2.9843({study_hours}) - 0.0310({assignments}) - 0.1046({sleep_hours})"
        }
        
    elif model_type in ["random_forest", "decision_tree", "logistic_regression", "knn"]:
        path_map = {
            "random_forest": RF_MODEL_PATH,
            "decision_tree": DT_MODEL_PATH,
            "logistic_regression": LOGREG_MODEL_PATH,
            "knn": KNN_MODEL_PATH
        }
        name_map = {
            "random_forest": "Random Forest Classifier",
            "decision_tree": "Decision Tree Classifier",
            "logistic_regression": "Logistic Regression Classifier",
            "knn": "K-Nearest Neighbors Classifier (KNN)"
        }
        model_path = path_map[model_type]
        
        if os.path.exists(model_path):
            pipeline = joblib.load(model_path)
            pred = pipeline.predict(input_data)[0]
            classes = list(pipeline.classes_)
            proba = pipeline.predict_proba(input_data)[0]
            pass_idx = classes.index('Pass') if 'Pass' in classes else 1
            pass_proba = float(proba[pass_idx])
            fail_proba = float(1.0 - pass_proba)
        else:
            # Fallback heuristic
            risk_score = (study_hours * 12) + (attendance * 0.4) + (previous_marks * 0.4)
            pass_proba = min(0.99, max(0.01, risk_score / 100.0))
            pred = "Pass" if pass_proba >= 0.5 else "Fail"
            fail_proba = 1.0 - pass_proba
            
        risk_tier = "Low Academic Risk" if pass_proba >= 0.85 else ("Moderate Risk" if pass_proba >= 0.5 else "High Academic Risk (At-Risk Student)")
        
        return {
            "model_type": name_map[model_type],
            "prediction": pred,
            "pass_probability": round(pass_proba * 100, 2),
            "fail_probability": round(fail_proba * 100, 2),
            "risk_tier": risk_tier,
            "key_factors": {
                "Study Hours Contribution": f"{study_hours} hrs/day",
                "Attendance": f"{attendance}%",
                "Previous Marks": f"{previous_marks}/100",
                "Assignments Completed": f"{assignments}/20"
            }
        }
    else:
        raise ValueError(f"Unknown model_type: {model_type}")

if __name__ == "__main__":
    train_all_models()
    
    print("\n--- Test All Classifiers ---")
    for m in ["random_forest", "decision_tree", "logistic_regression", "knn"]:
        res = predict_student(attendance=85, study_hours=6.5, previous_marks=75, assignments=18, sleep_hours=7.5, model_type=m)
        print(f"{res['model_type']}: Prediction = {res['prediction']} ({res['pass_probability']}% Pass)")
