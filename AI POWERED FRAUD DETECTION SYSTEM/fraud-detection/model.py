import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve, precision_recall_curve, average_precision_score
)
from sklearn.pipeline import Pipeline
import pickle
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "model_artifacts.pkl")

CATEGORICAL_COLS = ["merchant_category", "card_type", "device_type", "country"]
NUMERIC_FEATURES = [
    "amount", "hour", "day_of_week", "distance_from_home_km",
    "is_foreign_transaction", "is_high_risk_country",
    "num_transactions_last_hour", "num_transactions_last_day",
    "ip_risk_score", "is_new_device", "account_age_days",
    "failed_attempts_last_hour", "amount_vs_user_avg",
]

def preprocess(df: pd.DataFrame, encoders: dict = None, fit: bool = True):
    df = df.copy()
    if encoders is None:
        encoders = {}
    for col in CATEGORICAL_COLS:
        if col not in df.columns:
            continue
        if fit:
            le = LabelEncoder()
            df[f"{col}_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        else:
            le = encoders[col]
            known = set(le.classes_)
            df[col] = df[col].astype(str).apply(lambda x: x if x in known else le.classes_[0])
            df[f"{col}_enc"] = le.transform(df[col])
    return df, encoders

def get_feature_cols():
    return NUMERIC_FEATURES + [f"{c}_enc" for c in CATEGORICAL_COLS]

def train_model(df: pd.DataFrame, model_type: str = "Random Forest"):
    df, encoders = preprocess(df, fit=True)
    feature_cols = get_feature_cols()
    X = df[feature_cols]
    y = df["is_fraud"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    model_map = {
        "Random Forest": RandomForestClassifier(
            n_estimators=200, max_depth=12, min_samples_leaf=2,
            class_weight="balanced", random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=200, learning_rate=0.05, max_depth=5,
            subsample=0.8, random_state=42
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=1000, class_weight="balanced", C=0.5, random_state=42
        ),
        "XGBoost (GBM)": GradientBoostingClassifier(
            n_estimators=300, learning_rate=0.03, max_depth=6,
            subsample=0.8, min_samples_leaf=3, random_state=42
        ),
    }

    clf = model_map[model_type]
    clf.fit(X_train_s, y_train)

    y_pred = clf.predict(X_test_s)
    y_prob = clf.predict_proba(X_test_s)[:, 1]

    # Anomaly detection layer
    iso = IsolationForest(contamination=0.05, random_state=42, n_jobs=-1)
    iso.fit(X_train_s)
    anomaly_scores = -iso.score_samples(X_test_s)  # higher = more anomalous

    # Cross-val
    cv_scores = cross_val_score(clf, X_train_s, y_train, cv=5, scoring="roc_auc")

    # All model comparison
    comparison = {}
    for name, m in model_map.items():
        m.fit(X_train_s, y_train)
        p = m.predict_proba(X_test_s)[:, 1]
        comparison[name] = {
            "roc_auc": roc_auc_score(y_test, p),
            "avg_precision": average_precision_score(y_test, p),
        }

    fpr, tpr, _ = roc_curve(y_test, y_prob)
    pr_precision, pr_recall, _ = precision_recall_curve(y_test, y_prob)

    metrics = {
        "report": classification_report(y_test, y_pred, output_dict=True),
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_prob),
        "avg_precision": average_precision_score(y_test, y_prob),
        "roc_curve": (fpr, tpr),
        "pr_curve": (pr_precision, pr_recall),
        "cv_scores": cv_scores,
        "model_comparison": comparison,
        "feature_importances": (
            dict(zip(feature_cols, clf.feature_importances_))
            if hasattr(clf, "feature_importances_") else None
        ),
        "anomaly_scores": anomaly_scores,
        "y_test": y_test.values,
        "y_prob": y_prob,
        "feature_cols": feature_cols,
    }

    with open(MODEL_PATH, "wb") as f:
        pickle.dump({
            "model": clf, "scaler": scaler,
            "encoders": encoders, "iso": iso,
            "feature_cols": feature_cols
        }, f)

    return clf, scaler, encoders, metrics, X_test, y_test

def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None, None, None
    with open(MODEL_PATH, "rb") as f:
        a = pickle.load(f)
    return a["model"], a["scaler"], a["encoders"], a.get("iso")

def predict_transaction(transaction: dict, model, scaler, encoders):
    df = pd.DataFrame([transaction])
    df, _ = preprocess(df, encoders=encoders, fit=False)
    feature_cols = get_feature_cols()
    # Fill missing numeric cols with 0
    for col in feature_cols:
        if col not in df.columns:
            df[col] = 0
    X = df[feature_cols]
    X_s = scaler.transform(X)
    prob = model.predict_proba(X_s)[0][1]
    pred = int(model.predict(X_s)[0])
    return pred, prob, X_s

def batch_predict(df: pd.DataFrame, model, scaler, encoders):
    df2, _ = preprocess(df, encoders=encoders, fit=False)
    feature_cols = get_feature_cols()
    for col in feature_cols:
        if col not in df2.columns:
            df2[col] = 0
    X = df2[feature_cols]
    X_s = scaler.transform(X)
    probs = model.predict_proba(X_s)[:, 1]
    preds = model.predict(X_s)
    return preds, probs
