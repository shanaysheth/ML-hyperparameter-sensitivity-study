import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


def prepare_data(path: str):
    df = pd.read_csv(path)
    df["target"] = (df["rings"] >= 10).astype(int)
    df = pd.get_dummies(df, columns=["sex"], drop_first=True)
    X = df.drop(columns=["rings", "target"])
    y = df["target"]
    return X, y


def build_pipeline(name: str, estimator, scaler: bool):
    steps = []
    if scaler:
        steps.append(("scaler", StandardScaler()))
    steps.append(("clf", estimator))
    return Pipeline(steps)


def evaluate_model(pipeline: Pipeline, X_train, X_test, y_train, y_test):
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    probas = pipeline.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1_score": f1_score(y_test, y_pred),
        "auc_roc": roc_auc_score(y_test, probas),
    }


def variance_from_cv(pipeline: Pipeline, X, y):
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scoring = {"accuracy": "accuracy", "f1_score": "f1", "auc_roc": "roc_auc"}
    cv_results = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, return_train_score=False)
    return {
        metric: np.var(cv_results[f"test_{metric}"])
        for metric in scoring.keys()
    }


def main():
    X, y = prepare_data("abalone_original.csv")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, stratify=y, test_size=0.25, random_state=42
    )

    models = {
        "knn": {
            "estimator": KNeighborsClassifier(n_neighbors=7, weights="distance"),
            "scaler": True,
            "hyperparams": {
                "n_neighbors": 7,
                "weights": "distance",
            },
        },
        "svm": {
            "estimator": SVC(C=1.5, kernel="rbf", probability=True, gamma="scale"),
            "scaler": True,
            "hyperparams": {
                "C": 1.5,
                "kernel": "rbf",
                "gamma": "scale",
            },
        },
        "decision_tree": {
            "estimator": DecisionTreeClassifier(max_depth=6, min_samples_leaf=3, random_state=42),
            "scaler": False,
            "hyperparams": {
                "max_depth": 6,
                "min_samples_leaf": 3,
            },
        },
        "logistic_regression": {
            "estimator": LogisticRegression(C=1.0, solver="liblinear", random_state=42),
            "scaler": True,
            "hyperparams": {
                "C": 1.0,
                "solver": "liblinear",
            },
        },
    }

    report = []
    for name, config in models.items():
        pipeline = build_pipeline(name, config["estimator"], config["scaler"])
        metrics = evaluate_model(pipeline, X_train, X_test, y_train, y_test)
        variances = variance_from_cv(pipeline, X_train, y_train)
        report.append(
            {
                "model": name,
                "metrics": metrics,
                "variance": variances,
                "hyperparams": config["hyperparams"],
            }
        )

    for entry in report:
        print(f"\nModel: {entry['model']}")
        for metric, value in entry["metrics"].items():
            print(f"  {metric:10}: {value:.4f}")
        print("  Variance per metric:")
        for metric, value in entry["variance"].items():
            print(f"    {metric:10}: {value:.6f}")
        print("  Hyperparameters:")
        for param, value in entry["hyperparams"].items():
            print(f"    {param:12}: {value}")


if __name__ == "__main__":
    main()
