#!/usr/bin/env python3
"""Send two example students to the API.

Usage:
    python test_predict.py                              # http://localhost:9696
    python test_predict.py https://<your-app>.fly.dev   # deployed service
"""
import json
import sys
from urllib.request import Request, urlopen

base = sys.argv[1].rstrip("/") if len(sys.argv) > 1 else "http://localhost:9696"

engaged = {
    "code_module": "BBB", "gender": "F", "region": "London Region",
    "highest_education": "HE Qualification", "imd_band": "60-70%", "age_band": "35-55",
    "disability": "N", "num_of_prev_attempts": 0, "studied_credits": 60,
    "date_registration": -45, "clicks_before_start": 150,
    "weekly_clicks": [120, 95, 110, 80, 130, 90, 105, 85],
    "weekly_active_days": [5, 4, 5, 4, 5, 4, 5, 4],
    "assessments_due": 2, "assessment_scores": [82, 76], "late_submissions": 0,
}
disengaged = {
    **engaged,
    "highest_education": "Lower Than A Level", "imd_band": "0-10%", "age_band": "0-35",
    "num_of_prev_attempts": 1, "studied_credits": 120, "clicks_before_start": 0,
    "weekly_clicks": [40, 15, 5, 0, 0, 3, 0, 0],
    "weekly_active_days": [2, 1, 1, 0, 0, 1, 0, 0],
    "assessment_scores": [], "late_submissions": 0,
}

with urlopen(f"{base}/health") as r:
    print("health:", json.load(r))

for name, student in [("engaged", engaged), ("disengaged", disengaged)]:
    req = Request(f"{base}/predict", data=json.dumps(student).encode(),
                  headers={"Content-Type": "application/json"}, method="POST")
    with urlopen(req) as r:
        print(f"{name:10s} ->", json.load(r))
