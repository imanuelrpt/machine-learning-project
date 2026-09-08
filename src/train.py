"""Pelatihan model end-to-end untuk Prediksi Penyakit Jantung.

Alur:
1. Muat & bersihkan data
2. Split train/test
3. Bandingkan 6 model dengan cross-validation (5-fold)
4. Hyperparameter tuning (GridSearchCV) untuk model teratas
5. Evaluasi model final pada test set
6. Simpan artefak (model, tabel perbandingan, figure ROC/confusion/feature importance)

Cara pakai:
    python -m src.train
"""

from __future__ import annotations

import json
import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    auc,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_curve,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier

from src.preprocessing import build_preprocessor, clean_data, load_data
from src.utils import (
    COLOR_LIST,
    COMPARISON_PATH,
    FIGURE_DIR,
    MODEL_INFO_PATH,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    PROJECT_ROOT,
    TARGET_COL,
    save_figure,
    style_plot,
)

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
CV_FOLDS = 5

# ==================== DAFTAR MODEL ====================
def build_models() -> dict[str, object]:
    """Daftar model yang dibandingkan."""
    return {
        "Logistic Regression": LogisticRegression(max_iter=2000, random_state=RANDOM_STATE),
        "Decision Tree": DecisionTreeClassifier(max_depth=5, random_state=RANDOM_STATE),
        "Random Forest": RandomForestClassifier(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
        "KNN": KNeighborsClassifier(n_neighbors=7),
        "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=300, max_depth=4, learning_rate=0.1,
            random_state=RANDOM_STATE, eval_metric="logloss",
        ),
    }


# ==================== CROSS-VALIDATION ====================
def evaluate_cv_models(X_train: pd.DataFrame, y_train: pd.Series) -> pd.DataFrame:
    """Evaluasi semua model dengan 5-fold cross-validation (metrik rata-rata)."""
    preprocessor = build_preprocessor()
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "accuracy": "accuracy",
        "precision": "precision",
        "recall": "recall",
        "f1": "f1",
        "roc_auc": "roc_auc",
    }

    rows = []
    for name, model in build_models().items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        scores = cross_validate(
            pipe, X_train, y_train, cv=cv, scoring=scoring, n_jobs=-1, return_train_score=False
        )
        rows.append(
            {
                "Model": name,
                "Accuracy": scores["test_accuracy"].mean(),
                "Precision": scores["test_precision"].mean(),
                "Recall": scores["test_recall"].mean(),
                "F1-score": scores["test_f1"].mean(),
                "ROC-AUC": scores["test_roc_auc"].mean(),
            }
        )
        print(f"  [CV] {name:<22} acc={scores['test_accuracy'].mean():.4f} "
              f"recall={scores['test_recall'].mean():.4f} roc={scores['test_roc_auc'].mean():.4f}")

    comparison = pd.DataFrame(rows).set_index("Model")
    comparison.to_csv(COMPARISON_PATH)
    return comparison


# ==================== GRID SEARCH ====================
def grid_search_best_model(
    X_train: pd.DataFrame, y_train: pd.Series
) -> tuple[str, GridSearchCV]:
    """GridSearchCV untuk model top-2 berdasarkan ROC-AUC CV."""
    grids = {
        "Random Forest": (
            RandomForestClassifier(random_state=RANDOM_STATE, n_jobs=-1),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 5, 8],
                "model__min_samples_split": [2, 5],
            },
        ),
        "XGBoost": (
            XGBClassifier(random_state=RANDOM_STATE, eval_metric="logloss"),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, 7],
                "model__learning_rate": [0.05, 0.1, 0.2],
            },
        ),
    }

    preprocessor = build_preprocessor()
    best_gs, best_score, best_name = None, -1.0, None
    for name, (estimator, param_grid) in grids.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        gs = GridSearchCV(
            pipe, param_grid, cv=CV_FOLDS, scoring="roc_auc",
            n_jobs=-1, verbose=0, error_score="raise",
        )
        gs.fit(X_train, y_train)
        print(f"  [GridSearch] {name:<15} best_roc_auc={gs.best_score_:.4f} params={gs.best_params_}")
        if gs.best_score_ > best_score:
            best_score, best_gs, best_name = gs.best_score_, gs, name

    return best_name, best_gs


# ==================== EVALUASI + FIGURES ====================
def evaluate_test(pipe, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipe.predict(X_test)
    y_proba = pipe.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": auc(*roc_curve(y_test, y_proba)[:2]),
    }
    return metrics, y_pred, y_proba


def plot_confusion_matrix(y_test, y_pred, name: str) -> None:
    fig, ax = plt.subplots(figsize=(5, 4))
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", cbar=False,
        xticklabels=["Sehat", "Berpenyakit"], yticklabels=["Sehat", "Berpenyakit"], ax=ax,
    )
    ax.set_xlabel("Prediksi")
    ax.set_ylabel("Aktual")
    ax.set_title(f"Confusion Matrix — {name}")
    save_figure(fig, "confusion_matrix.png")


def plot_roc_curves(pipes: dict[str, object], X_test: pd.DataFrame, y_test: pd.Series) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for (name, pipe), color in zip(pipes.items(), COLOR_LIST):
        y_proba = pipe.predict_proba(X_test)[:, 1]
        fpr, tpr, _ = roc_curve(y_test, y_proba)
        ax.plot(fpr, tpr, color=color, lw=2, label=f"{name} (AUC={auc(fpr, tpr):.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=1, label="Acak (AUC=0.5)")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("Kurva ROC — Semua Model")
    ax.legend(loc="lower right", fontsize=8)
    save_figure(fig, "roc_curves.png")


def plot_feature_importance(pipe, feature_names: list[str], name: str) -> None:
    model = pipe.named_steps["model"]
    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif isinstance(model, LogisticRegression):
        importance = np.abs(model.coef_[0])
    else:
        print("  [!] Model tidak mendukung feature_importance, dilewati.")
        return

    imp = pd.DataFrame({"feature": feature_names, "importance": importance})
    imp = imp.sort_values("importance", ascending=False).head(15)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=imp, y="feature", x="importance", palette="viridis", ax=ax)
    ax.set_title(f"Feature Importance (Permutation/Importance) — {name}")
    ax.set_xlabel("Kepentingan relatif")
    ax.set_ylabel("")
    save_figure(fig, "feature_importance.png")


def print_markdown_table(comparison: pd.DataFrame) -> str:
    md = comparison.round(4).to_markdown() if hasattr(comparison, "to_markdown") else ""
    if not md:
        md = comparison.round(4).to_string()
    return md


# ==================== MAIN ====================
def main() -> None:
    style_plot()
    print("=" * 70)
    print("PIPELINE TRAINING: PREDIKSI PENYAKIT JANTUNG (Scikit-Learn)")
    print("=" * 70)

    # 1. Muat & bersihkan data
    df_raw = load_data()
    df = clean_data(df_raw)
    print(f"\n[1] Data dimuat: {df.shape[0]} baris, {df.shape[1]} kolom")

    X = df.drop(columns=[TARGET_COL])
    y = df[TARGET_COL]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )
    print(f"    Train: {X_train.shape[0]} | Test: {X_test.shape[0]}")
    print(f"    Distribusi target: sehat={sum(y==0)} ({y.mean()*100:.0f}% positive)")

    # 2. Perbandingan model dengan CV
    print("\n[2] Cross-Validation (5-fold) — perbandingan 6 model:")
    comparison = evaluate_cv_models(X_train, y_train)
    print("\n    Tabel perbandingan rata-rata CV:")
    print(print_markdown_table(comparison))

    # 3. Grid search
    print("\n[3] Hyperparameter Tuning (GridSearchCV, scoring=ROC-AUC):")
    best_name, best_gs = grid_search_best_model(X_train, y_train)
    best_pipe = best_gs.best_estimator_
    print(f"    Model terbaik: {best_name} | score grid = {best_gs.best_score_:.4f}")

    # 4. Evaluasi test set
    metrics, y_pred, _ = evaluate_test(best_pipe, X_test, y_test)
    print("\n[4] Evaluasi model terbaik pada TEST SET:")
    print(f"    Model       : {best_name}")
    for k, v in metrics.items():
        print(f"    {k:<10}: {v:.4f}")

    # 5. Simpan artefak
    print("\n[5] Menyimpan artefak...")
    joblib.dump(best_pipe, MODEL_PATH)
    joblib.dump(build_preprocessor().fit(X_train), PREPROCESSOR_PATH)
    model_info = {
        "model": best_name,
        "best_params": best_gs.best_params_,
        "test_metrics": metrics,
        "features": list(X.columns),
        "random_state": RANDOM_STATE,
    }
    with open(MODEL_INFO_PATH, "w") as f:
        json.dump(model_info, f, indent=2, default=str)

    # 6. Figures
    print("\n[6] Membuat figure evaluasi → reports/figures/")
    plot_confusion_matrix(y_test, y_pred, best_name)
    pipes = {}
    preprocessor = build_preprocessor().fit(X_train)
    for name, model in build_models().items():
        pipe = Pipeline([("prep", preprocessor), ("model", model)])
        pipe.fit(X_train, y_train)
        pipes[name] = pipe
    plot_roc_curves(pipes, X_test, y_test)
    plot_feature_importance(best_pipe, best_pipe.named_steps["prep"].get_feature_names_out(), best_name)

    # Figure perbandingan CV
    fig, ax = plt.subplots(figsize=(8, 5))
    comparison[["ROC-AUC", "F1-score", "Recall"]].plot.bar(ax=ax, rot=45, colormap="viridis")
    ax.set_title("Metrik Model (rata-rata 5-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.legend(loc="lower right")
    ax.set_ylabel("Skor")
    save_figure(fig, "model_comparison.png")

    print("\n" + "=" * 70)
    print(f"SELESAI. Model terbaik: {best_name} (test ROC-AUC={metrics['roc_auc']:.3f})")
    print("Artefak tersimpan di models/ dan reports/figures/")
    print("=" * 70)


if __name__ == "__main__":
    main()