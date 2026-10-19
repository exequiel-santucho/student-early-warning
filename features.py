"""Feature engineering shared by the notebook, train.py and predict.py.

A *raw record* describes what a tutor knows about a student at the cut-off week:
demographics, weekly activity in the virtual learning environment (VLE) and the
assessments submitted so far. `compute_features` turns it into model features.
Using the same function for training and serving avoids training/serving skew.
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

CUTOFF_WEEK = 8  # default: predict with the first 8 weeks (~20% of a presentation)
AT_RISK = {"Fail", "Withdrawn"}

CATEGORICAL = ["code_module", "gender", "region", "highest_education", "imd_band",
               "age_band", "disability"]


def _clean_category(value: Any) -> str:
    # OULAD marks some missing values with "?"
    if value is None or (isinstance(value, float) and np.isnan(value)) or value in ("", "?"):
        return "unknown"
    return str(value)


def compute_features(raw: dict[str, Any]) -> dict[str, Any]:
    """Raw record -> flat feature dict (input for DictVectorizer)."""
    clicks = np.asarray(raw.get("weekly_clicks") or [0], dtype=float)
    days = np.asarray(raw.get("weekly_active_days") or [0], dtype=float)
    scores = [s for s in (raw.get("assessment_scores") or []) if s is not None]
    n_weeks = len(clicks)
    log_clicks = np.log1p(clicks)
    active = np.flatnonzero(clicks > 0)

    features: dict[str, Any] = {c: _clean_category(raw.get(c)) for c in CATEGORICAL}

    n_due = int(raw.get("assessments_due") or 0)
    n_submitted = len(scores)
    features.update({
        "num_of_prev_attempts": int(raw.get("num_of_prev_attempts") or 0),
        "studied_credits": float(raw.get("studied_credits") or 0),
        "date_registration": float(raw.get("date_registration") or 0),
        # activity (counts are log-transformed: long tails)
        "clicks_before_start_log": float(np.log1p(raw.get("clicks_before_start") or 0)),
        "clicks_total_log": float(np.log1p(clicks.sum())),
        "clicks_first2_log": float(np.log1p(clicks[:2].sum())),
        "clicks_last2_log": float(np.log1p(clicks[-2:].sum())),
        "active_days": float(days.sum()),
        "active_weeks": int((clicks > 0).sum()),
        "weeks_since_last_activity": int(n_weeks - 1 - active[-1]) if len(active) else n_weeks,
        "clicks_trend": float(np.polyfit(np.arange(n_weeks), log_clicks, 1)[0]) if n_weeks > 1 else 0.0,
        # assessments
        "assessments_due": n_due,
        "assessments_submitted": n_submitted,
        "assessments_missed": max(n_due - n_submitted, 0),
        "late_submissions": int(raw.get("late_submissions") or 0),
        "score_mean": float(np.mean(scores)) if scores else 0.0,
        "score_min": float(np.min(scores)) if scores else 0.0,
    })
    return features


def build_raw_records(students: pd.DataFrame, assessments: pd.DataFrame,
                      submissions: pd.DataFrame, cutoff_week: int = CUTOFF_WEEK
                      ) -> tuple[list[dict[str, Any]], np.ndarray, pd.DataFrame]:
    """Build raw records + target for every student still enrolled at the cut-off.

    Students who unregistered before the cut-off are excluded: at that point
    there is nothing left to predict.
    """
    cutoff_day = cutoff_week * 7
    keys = ["code_module", "code_presentation", "id_student"]

    df = students[~(students.date_unregistration < cutoff_day)].copy()

    due = (assessments[assessments.date < cutoff_day]
           .groupby(["code_module", "code_presentation"]).size().rename("assessments_due"))
    df = df.merge(due, on=["code_module", "code_presentation"], how="left")
    df["assessments_due"] = df.assessments_due.fillna(0).astype(int)

    subs = submissions[submissions.date_submitted < cutoff_day].copy()
    subs["late"] = (subs.date_submitted > subs.date_due).astype(int)
    agg = subs.groupby(keys).agg(scores=("score", list), late_submissions=("late", "sum"))
    df = df.merge(agg, on=keys, how="left")

    clicks_cols = [f"clicks_w{w}" for w in range(cutoff_week)]
    days_cols = [f"days_w{w}" for w in range(cutoff_week)]

    records = []
    for row in df.itertuples(index=False):
        r = row._asdict()
        scores = r["scores"] if isinstance(r["scores"], list) else []
        records.append({
            **{c: r[c] for c in CATEGORICAL},
            "num_of_prev_attempts": r["num_of_prev_attempts"],
            "studied_credits": r["studied_credits"],
            "date_registration": None if pd.isna(r["date_registration"]) else r["date_registration"],
            "clicks_before_start": r["clicks_before_start"],
            "weekly_clicks": [r[c] for c in clicks_cols],
            "weekly_active_days": [r[c] for c in days_cols],
            "assessments_due": r["assessments_due"],
            "assessment_scores": [None if pd.isna(s) else float(s) for s in scores],
            "late_submissions": 0 if pd.isna(r["late_submissions"]) else int(r["late_submissions"]),
        })

    y = df.final_result.isin(AT_RISK).astype(int).values
    meta = df[keys + ["final_result"]].reset_index(drop=True)
    return records, y, meta


def load_tables(data_dir) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (pd.read_csv(f"{data_dir}/students.csv"),
            pd.read_csv(f"{data_dir}/assessments.csv"),
            pd.read_csv(f"{data_dir}/student_assessments.csv"))
