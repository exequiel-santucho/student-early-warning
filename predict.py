"""FastAPI service: risk of failing or withdrawing, given the first weeks of a course.

Run:
    uv run uvicorn predict:app --host 0.0.0.0 --port 9696
Docs: http://localhost:9696/docs
"""
from __future__ import annotations

import json
import os
import pickle
from pathlib import Path
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field

from features import CUTOFF_WEEK, compute_features

MODEL_DIR = Path(os.getenv("MODEL_DIR", Path(__file__).with_name("models")))
with (MODEL_DIR / "model.bin").open("rb") as f:
    pipeline = pickle.load(f)
METADATA = json.loads((MODEL_DIR / "metadata.json").read_text(encoding="utf-8"))
THRESHOLD = METADATA["threshold"]
HIGH_RISK = 0.75

app = FastAPI(
    title="Student early-warning API",
    description=f"Predicts the probability that a student fails or withdraws, "
                f"using demographics and activity during the first {CUTOFF_WEEK} weeks.",
    version="1.0.0",
)


class Student(BaseModel):
    code_module: str = Field(examples=["BBB"], description="Module code (AAA-GGG)")
    gender: str | None = Field(None, examples=["F"])
    region: str | None = Field(None, examples=["London Region"])
    highest_education: str | None = Field(None, examples=["A Level or Equivalent"])
    imd_band: str | None = Field(None, examples=["20-30%"], description="Deprivation index band")
    age_band: str | None = Field(None, examples=["0-35"])
    disability: str | None = Field(None, examples=["N"])
    num_of_prev_attempts: int = Field(0, ge=0)
    studied_credits: float = Field(60, ge=0)
    date_registration: float | None = Field(None, description="Days relative to course start")
    clicks_before_start: int = Field(0, ge=0)
    weekly_clicks: list[int] = Field(min_length=CUTOFF_WEEK, max_length=CUTOFF_WEEK,
                                     description=f"VLE clicks in weeks 0..{CUTOFF_WEEK - 1}")
    weekly_active_days: list[int] = Field(min_length=CUTOFF_WEEK, max_length=CUTOFF_WEEK)
    assessments_due: int = Field(0, ge=0, description="Assessments due before the cut-off")
    assessment_scores: list[float] = Field(default_factory=list, description="Scores of submitted assessments")
    late_submissions: int = Field(0, ge=0)


class Prediction(BaseModel):
    at_risk_probability: float
    at_risk: bool
    risk_level: Literal["low", "medium", "high"]


@app.get("/health")
def health() -> dict:
    return {"status": "ok", "model_sha256": METADATA["model_sha256"],
            "cutoff_week": CUTOFF_WEEK, "threshold": THRESHOLD}


@app.post("/predict", response_model=Prediction)
def predict(student: Student) -> Prediction:
    features = compute_features(student.model_dump())
    p = float(pipeline.predict_proba([features])[0, 1])
    level = "high" if p >= HIGH_RISK else "medium" if p >= THRESHOLD else "low"
    return Prediction(at_risk_probability=round(p, 4), at_risk=p >= THRESHOLD, risk_level=level)
