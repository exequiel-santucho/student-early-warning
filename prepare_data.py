#!/usr/bin/env python3
"""Download OULAD and build the compact tables used by the project.

The raw click log (studentVle.csv) has ~10M rows / 450 MB, so we aggregate it
into weekly totals once and save small CSVs under data/:

- data/students.csv            one row per student enrolment (module + presentation),
                               demographics, registration, final result, and
                               clicks / active days for weeks 0..19
- data/assessments.csv         assessment schedule (non-exam) per module presentation
- data/student_assessments.csv submissions with score and submission day

Usage:
    uv run python prepare_data.py            # downloads the zip if needed
"""
from __future__ import annotations

import zipfile
from pathlib import Path
from urllib.request import urlretrieve

import numpy as np
import pandas as pd

URL = "https://archive.ics.uci.edu/static/public/349/open+university+learning+analytics+dataset.zip"
HERE = Path(__file__).resolve().parent
RAW = HERE / "data" / "raw"
OUT = HERE / "data"
N_WEEKS = 20  # store 20 weeks so the notebook can try different cut-off points


def download() -> None:
    if (RAW / "studentVle.csv").exists():
        return
    RAW.mkdir(parents=True, exist_ok=True)
    zip_path = RAW / "oulad.zip"
    if not zip_path.exists():
        print(f"Downloading {URL} ...")
        urlretrieve(URL, zip_path)
    with zipfile.ZipFile(zip_path) as zf:
        zf.extractall(RAW)


def weekly_activity() -> pd.DataFrame:
    """Aggregate the click log into clicks and active days per week."""
    keys = ["code_module", "code_presentation", "id_student"]
    vle = pd.read_csv(
        RAW / "studentVle.csv",
        usecols=keys + ["date", "sum_click"],
        dtype={"code_module": "category", "code_presentation": "category",
               "id_student": "int32", "date": "int16", "sum_click": "int32"},
    )

    before = (vle[vle.date < 0].groupby(keys, observed=True).sum_click.sum()
              .rename("clicks_before_start"))

    vle = vle[(vle.date >= 0) & (vle.date < N_WEEKS * 7)].copy()
    vle["week"] = (vle.date // 7).astype("int8")

    clicks = vle.pivot_table(index=keys, columns="week", values="sum_click",
                             aggfunc="sum", fill_value=0, observed=True)
    clicks.columns = [f"clicks_w{w}" for w in clicks.columns]

    days = (vle.groupby(keys + ["week"], observed=True).date.nunique()
            .unstack(fill_value=0))
    days.columns = [f"days_w{w}" for w in days.columns]

    activity = clicks.join(days, how="outer").join(before, how="outer").fillna(0)
    return activity.astype("int32").reset_index()


def main() -> None:
    download()

    info = pd.read_csv(RAW / "studentInfo.csv")
    reg = pd.read_csv(RAW / "studentRegistration.csv", na_values="?")
    courses = pd.read_csv(RAW / "courses.csv")

    keys = ["code_module", "code_presentation", "id_student"]
    students = info.merge(reg, on=keys, how="left").merge(
        courses, on=["code_module", "code_presentation"], how="left")

    print("Aggregating click log (takes ~1 min)...")
    activity = weekly_activity()
    for col in ("code_module", "code_presentation"):
        activity[col] = activity[col].astype(str)
    students = students.merge(activity, on=keys, how="left")

    activity_cols = [c for c in students.columns if c.startswith(("clicks_", "days_"))]
    students[activity_cols] = students[activity_cols].fillna(0).astype(int)
    # make sure every week column exists, even if nobody clicked that week
    for w in range(N_WEEKS):
        for prefix in ("clicks_w", "days_w"):
            if f"{prefix}{w}" not in students:
                students[f"{prefix}{w}"] = 0

    ordered = (["code_module", "code_presentation", "id_student", "gender", "region",
                "highest_education", "imd_band", "age_band", "num_of_prev_attempts",
                "studied_credits", "disability", "date_registration",
                "date_unregistration", "module_presentation_length", "final_result",
                "clicks_before_start"]
               + [f"clicks_w{w}" for w in range(N_WEEKS)]
               + [f"days_w{w}" for w in range(N_WEEKS)])
    students = students[ordered]
    students.to_csv(OUT / "students.csv", index=False)

    assessments = pd.read_csv(RAW / "assessments.csv", na_values="?")
    assessments = assessments[assessments.assessment_type != "Exam"]
    assessments.to_csv(OUT / "assessments.csv", index=False)

    submissions = pd.read_csv(RAW / "studentAssessment.csv", na_values="?")
    submissions = submissions.merge(
        assessments[["id_assessment", "code_module", "code_presentation", "date"]],
        on="id_assessment", how="inner").rename(columns={"date": "date_due"})
    submissions.to_csv(OUT / "student_assessments.csv", index=False)

    print(f"students.csv            {students.shape}")
    print(f"assessments.csv         {assessments.shape}")
    print(f"student_assessments.csv {submissions.shape}")
    print(students.final_result.value_counts(normalize=True).round(3))


if __name__ == "__main__":
    main()
