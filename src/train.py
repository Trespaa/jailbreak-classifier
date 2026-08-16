"""
train.py

Entrena y compara dos clasificadores base sobre características TF-IDF
(word 1-2 grams + char 3-5 grams, combinados):

  1. Logistic Regression  -- lineal, probabilidades calibradas, barato
  2. Linear SVM           -- a menudo más fuerte en texto disperso de alta dimensión

Ejecutar:
    python src/train.py
"""

import json
import joblib
import numpy as np
import pandas as pd
from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score,
    precision_recall_fscore_support, accuracy_score
)
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
MODELS = ROOT / "models"
FIGS = ROOT / "reports" / "figures"
MODELS.mkdir(exist_ok=True)
FIGS.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 42


def load_split(name):
    df = pd.read_csv(DATA / f"{name}.csv")
    return df["prompt"].astype(str), df["label"].values


def build_vectorizer():
    word_vec = TfidfVectorizer(
        analyzer="word", ngram_range=(1, 2),
        min_df=2, max_df=0.9, sublinear_tf=True
    )
    char_vec = TfidfVectorizer(
        analyzer="char_wb", ngram_range=(3, 5),
        min_df=2, max_df=0.9, sublinear_tf=True
    )
    return FeatureUnion([("word", word_vec), ("char", char_vec)])


def evaluate(name, y_true, y_pred, y_score):
    acc = accuracy_score(y_true, y_pred)
    p, r, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, average="binary"
    )
    auc = roc_auc_score(y_true, y_score)
    print(f"\n--- {name} ---")
    print(classification_report(y_true, y_pred, target_names=["regular", "jailbreak"]))
    print(f"ROC-AUC: {auc:.4f}")

    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["regular", "jailbreak"])
    ax.set_yticks([0, 1]); ax.set_yticklabels(["regular", "jailbreak"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_title(f"Confusion matrix -- {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im)
    fig.tight_layout()
    fig.savefig(FIGS / f"confusion_{name.lower().replace(' ', '_')}.png", dpi=150)
    plt.close(fig)

    return {
        "accuracy": acc, "precision": p, "recall": r,
        "f1": f1, "roc_auc": auc, "confusion_matrix": cm.tolist()
    }


def main():
    X_train, y_train = load_split("train")
    X_val, y_val = load_split("val")
    X_test, y_test = load_split("test")

    results = {}

    logreg = Pipeline([
        ("features", build_vectorizer()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced",
                                    random_state=RANDOM_STATE)),
    ])
    logreg.fit(X_train, y_train)
    val_pred = logreg.predict(X_val)
    val_score = logreg.predict_proba(X_val)[:, 1]
    results["logreg_val"] = evaluate("LogReg (val)", y_val, val_pred, val_score)

    svm = Pipeline([
        ("features", build_vectorizer()),
        ("clf", CalibratedClassifierCV(
            LinearSVC(class_weight="balanced", random_state=RANDOM_STATE),
            cv=3
        )),
    ])
    svm.fit(X_train, y_train)
    val_pred_svm = svm.predict(X_val)
    val_score_svm = svm.predict_proba(X_val)[:, 1]
    results["svm_val"] = evaluate("LinearSVM (val)", y_val, val_pred_svm, val_score_svm)

    best_name, best_model = (
        ("logreg", logreg) if results["logreg_val"]["f1"] >= results["svm_val"]["f1"]
        else ("svm", svm)
    )
    print(f"\nSelected model based on validation F1: {best_name}")

    test_pred = best_model.predict(X_test)
    test_score = best_model.predict_proba(X_test)[:, 1]
    results["test_final"] = evaluate(f"{best_name} (TEST - held out)", y_test, test_pred, test_score)
    results["selected_model"] = best_name

    joblib.dump(best_model, MODELS / "jailbreak_classifier.joblib")
    with open(ROOT / "reports" / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    test_df = pd.read_csv(DATA / "test.csv").reset_index(drop=True)
    test_df["pred"] = test_pred
    test_df["score"] = test_score
    errors = test_df[test_df["label"] != test_df["pred"]]
    errors.to_csv(ROOT / "reports" / "test_errors.csv", index=False)
    print(f"\n{len(errors)} misclassified test examples saved to reports/test_errors.csv")
    print(f"Best model saved to models/jailbreak_classifier.joblib")


if __name__ == "__main__":
    main()