from pathlib import Path
import json
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
    confusion_matrix, classification_report
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DEFAULT_TARGET = "Attrition"
FEATURE_DROP_CANDIDATES = ["EmployeeNumber", "EmployeeCount", "Over18", "StandardHours"]

def load_dataset(path_or_dir):
    """Load a CSV from a file or find a CSV in the supplied data directory."""
    p = Path(path_or_dir)
    if p.is_file():
        return pd.read_csv(p)
    if p.is_dir():
        preferred = [
            p / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
            p / "IBM_HR_Analytics.csv",
            p / "HR_Analytics.csv",
            p / "WA_Fn-UseC_-HR-Employee-Attrition" / "WA_Fn-UseC_-HR-Employee-Attrition.csv",
        ]
        for candidate in preferred:
            if candidate.exists():
                return pd.read_csv(candidate)
        csv_files = sorted(p.glob("*.csv"))
        if csv_files:
            return pd.read_csv(csv_files[0])
        raise FileNotFoundError(f"No CSV file found in {p}. Put the IBM HR CSV in this folder.")
    raise FileNotFoundError(f"Dataset path does not exist: {p}")

def get_feature_columns(df, target=DEFAULT_TARGET):
    if target not in df.columns:
        raise ValueError(f"Target column '{target}' is missing.")
    excluded = {target, *FEATURE_DROP_CANDIDATES}
    # Exclude fully constant columns, too.
    for col in df.columns:
        if df[col].nunique(dropna=False) <= 1:
            excluded.add(col)
    features = [c for c in df.columns if c not in excluded]
    if not features:
        raise ValueError("No usable feature columns remain after excluding target/ID/constant columns.")
    return features

def encode_target(series):
    raw = series.astype(str).str.strip()
    lower = raw.str.lower()
    known = {"yes": 1, "true": 1, "1": 1, "left": 1, "attrition": 1,
             "no": 0, "false": 0, "0": 0, "stayed": 0, "stay": 0}
    mapped = lower.map(known)
    if mapped.notna().all() and mapped.nunique() == 2:
        return mapped.astype(int)
    unique = sorted(raw.dropna().unique().tolist())
    if len(unique) != 2:
        raise ValueError(f"Expected a binary target with 2 classes; found {len(unique)}: {unique}")
    # For non-standard labels, use the second sorted class as positive.
    mapping = {unique[0]: 0, unique[1]: 1}
    return raw.map(mapping).astype(int)

def build_pipeline(X, classifier):
    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]
    numeric_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])
    categorical_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])
    transformers = []
    if numeric_features:
        transformers.append(("numeric", numeric_pipe, numeric_features))
    if categorical_features:
        transformers.append(("categorical", categorical_pipe, categorical_features))
    preprocessor = ColumnTransformer(transformers=transformers, remainder="drop")
    return Pipeline([("preprocessor", preprocessor), ("classifier", classifier)])

def _metrics(name, model, X_test, y_test):
    pred = model.predict(X_test)
    if hasattr(model, "predict_proba"):
        scores = model.predict_proba(X_test)[:, 1]
        auc = float(roc_auc_score(y_test, scores))
    else:
        scores = pred
        auc = float("nan")
    return {
        "model": name,
        "accuracy": float(accuracy_score(y_test, pred)),
        "precision": float(precision_score(y_test, pred, zero_division=0)),
        "recall": float(recall_score(y_test, pred, zero_division=0)),
        "f1": float(f1_score(y_test, pred, zero_division=0)),
        "roc_auc": auc,
        "confusion_matrix": confusion_matrix(y_test, pred, labels=[0, 1]).tolist(),
        "classification_report": classification_report(y_test, pred, labels=[0, 1], target_names=["Stayed/No", "Left/Yes"], output_dict=True, zero_division=0),
    }

def train_and_save(df=None, root_dir=None, dataset_path=None, target=DEFAULT_TARGET):
    root = Path(root_dir) if root_dir else Path(__file__).resolve().parent
    model_dir = root / "models"
    report_dir = root / "reports"
    data_dir = root / "data"
    model_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    if df is None:
        df = load_dataset(dataset_path or data_dir)
    df = df.copy()
    df = df.dropna(subset=[target])
    feature_columns = get_feature_columns(df, target)
    X = df[feature_columns].copy()
    y = encode_target(df[target])
    if y.nunique() != 2:
        raise ValueError("The target must contain both classes (Stayed/No and Left/Yes).")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )
    models = {
        "Logistic Regression": LogisticRegression(
            max_iter=2000, class_weight="balanced", random_state=42
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=350, min_samples_leaf=2, class_weight="balanced",
            random_state=42, n_jobs=-1
        ),
    }
    fitted = {}
    metric_rows = []
    reports = {}
    for name, classifier in models.items():
        pipeline = build_pipeline(X_train, classifier)
        pipeline.fit(X_train, y_train)
        result = _metrics(name, pipeline, X_test, y_test)
        metric_rows.append({k: result[k] for k in ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]})
        reports[name] = {
            "classification_report": result["classification_report"],
            "confusion_matrix": result["confusion_matrix"],
        }
        fitted[name] = (pipeline, result)
    # Select on F1 to account for the minority attrition class, not accuracy alone.
    best_name = max(fitted, key=lambda name: fitted[name][1]["f1"])
    best_model, best_result = fitted[best_name]
    joblib.dump(best_model, model_dir / "hr_attrition_pipeline.pkl")
    pd.DataFrame(metric_rows).sort_values("f1", ascending=False).to_csv(report_dir / "model_comparison.csv", index=False)
    (report_dir / "classification_reports.json").write_text(json.dumps(reports, indent=2), encoding="utf-8")
    metadata = {
        "best_model": best_name,
        "target": target,
        "feature_columns": feature_columns,
        "rows": int(len(df)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_label": "Attrition Yes / Left",
        "selection_metric": "F1-score",
        "random_state": 42,
        "note": "Model estimates are not suitable for automated employment decisions."
    }
    (model_dir / "model_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    # Feature importance for the saved best model.
    try:
        preprocessor = best_model.named_steps["preprocessor"]
        names = preprocessor.get_feature_names_out()
        classifier = best_model.named_steps["classifier"]
        if hasattr(classifier, "feature_importances_"):
            importance = classifier.feature_importances_
        elif hasattr(classifier, "coef_"):
            importance = np.abs(classifier.coef_[0])
        else:
            importance = None
        if importance is not None and len(names) == len(importance):
            imp = pd.DataFrame({"feature": names, "importance": importance}).sort_values("importance", ascending=False)
            imp.to_csv(report_dir / "feature_importance.csv", index=False)
    except Exception as exc:
        warnings.warn(f"Could not create feature importance report: {exc}")
    return {
        "best_model": best_name,
        "metrics": metric_rows,
        "best_confusion_matrix": best_result["confusion_matrix"],
        "feature_columns": feature_columns,
    }

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent
    dataset = load_dataset(project_root / "data")
    result = train_and_save(dataset, root_dir=project_root)
    print("Training complete.")
    print("Best model:", result["best_model"])
    print(pd.DataFrame(result["metrics"]).sort_values("f1", ascending=False).to_string(index=False))
    print("Saved model: models/hr_attrition_pipeline.pkl")
    print("Saved reports: reports/")
