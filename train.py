#!/usr/bin/env python3
"""Train the final early-warning model and save it to models/.

1. Fit on train presentations (2013B, 2013J) and pick the decision threshold
   that maximizes F1 on the validation presentation (2014B).
2. Re-fit on train + validation and report AUC on the test presentation (2014J).
3. Save the scikit-learn pipeline (DictVectorizer + StandardScaler + LogisticRegression)
   and its metadata.

Usage:
    uv run python train.py
"""
from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
import sklearn
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score, precision_recall_curve, roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from features import CUTOFF_WEEK, build_raw_records, compute_features, load_tables

HERE = Path(__file__).resolve().parent

# Selected in notebook.ipynb (section 5): logistic regression tied with tuned XGBoost
# on validation (AUC 0.8255 vs 0.8254) and is simpler to explain, so it is the final model.
LR_PARAMS = {"C": 0.1, "max_iter": 2000}
TRAIN, VAL, TEST = ["2013B", "2013J"], ["2014B"], ["2014J"]


def build_pipeline():
    return make_pipeline(DictVectorizer(sparse=False), StandardScaler(), LogisticRegression(**LR_PARAMS))


def main(data_dir: Path, model_dir: Path) -> None:
    students, assessments, submissions = load_tables(data_dir)
    records, y, meta = build_raw_records(students, assessments, submissions, CUTOFF_WEEK)
    X = np.array([compute_features(r) for r in records], dtype=object)
    pres = meta.code_presentation.values
    tr, va, te = np.isin(pres, TRAIN), np.isin(pres, VAL), np.isin(pres, TEST)

    # 1) threshold on validation
    model = build_pipeline().fit(list(X[tr]), y[tr])
    p_val = model.predict_proba(list(X[va]))[:, 1]
    precision, recall, thresholds = precision_recall_curve(y[va], p_val)
    f1 = 2 * precision * recall / (precision + recall + 1e-12)
    threshold = float(round(thresholds[np.argmax(f1[:-1])], 3))
    val_auc = roc_auc_score(y[va], p_val)

    # 2) final model on train + validation, evaluated once on test
    full = tr | va
    model = build_pipeline().fit(list(X[full]), y[full])
    p_test = model.predict_proba(list(X[te]))[:, 1]
    test_auc = roc_auc_score(y[te], p_test)
    test_f1 = f1_score(y[te], p_test >= threshold)

    model_dir.mkdir(exist_ok=True)
    model_path = model_dir / "model.bin"
    with model_path.open("wb") as f:
        pickle.dump(model, f, protocol=pickle.HIGHEST_PROTOCOL)

    metadata = {
        "model": "DictVectorizer + StandardScaler + LogisticRegression",
        "cutoff_week": CUTOFF_WEEK,
        "target": "final_result in {Fail, Withdrawn}",
        "train_presentations": TRAIN + VAL,
        "test_presentation": TEST[0],
        "params": LR_PARAMS,
        "threshold": threshold,
        "validation_auc": round(val_auc, 4),
        "test_auc": round(test_auc, 4),
        "test_f1": round(test_f1, 4),
        "n_train": int(full.sum()),
        "n_test": int(te.sum()),
        "scikit_learn": sklearn.__version__,
        "model_sha256": hashlib.sha256(model_path.read_bytes()).hexdigest(),
    }
    (model_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=HERE / "data")
    parser.add_argument("--model-dir", type=Path, default=HERE / "models")
    args = parser.parse_args()
    main(args.data_dir, args.model_dir)
