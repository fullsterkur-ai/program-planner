from __future__ import annotations
from state_store import (
    load_adaptive_state,
    save_adaptive_state,
    load_program_state,
    save_program_state,
)
from feedback_service import (
    record_workout_feedback,
    complete_week,
)
from planner import make_plan
from pathlib import Path
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from schemas import (
    AthleteInput,
    WorkoutFeedback,
    CompleteWeekRequest,
)
from dataset import (
    candidate_rows,
    load_dataset,
    load_ml_bundle,
    normalize,
    summarize_exercises,
)
from periodization import (
    choose_action,
    progress_load,
    make_week_schedule,
)
import json
ROOT = Path(__file__).resolve().parent
DATA_PATH = (  ROOT / "data" / "raw" / "programs_detailed_boostcamp_kaggle.csv")
FEEDBACK_PATH = ROOT / "data" / "processed" / "workout_feedback.csv"
TRAINING_DASHBOARD_PATH = ROOT / "models" / "training_dashboard.json"
app = FastAPI(title="Periodization Engine API", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["http://localhost:5173", "http://127.0.0.1:5173", "http://localhost:3000", "http://127.0.0.1:3000"], allow_credentials=True, allow_methods=["*"], allow_headers=["*"])

@app.get("/health")
def health(): return {"status": "ok", "dataset_path": str(DATA_PATH), "dataset_exists": DATA_PATH.exists()}
@app.get("/training-dashboard")
def get_training_dashboard():
    if not TRAINING_DASHBOARD_PATH.exists():
        raise HTTPException(
            status_code=404,
            detail=(
                "Training dashboard data not found. "
                "Run train_model.py to generate the training results."
            ),
        )

    try:
        with TRAINING_DASHBOARD_PATH.open(
            "r",
            encoding="utf-8",
        ) as file:
            data = json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        raise HTTPException(
            status_code=500,
            detail="Could not read the saved training dashboard data.",
        ) from exc

    return data
@app.post("/generate-plan")
def generate_plan(athlete: AthleteInput): return make_plan(athlete)

@app.post("/workout-feedback")
def workout_feedback_endpoint(feedback: WorkoutFeedback):
    return record_workout_feedback(feedback)

@app.post("/complete-week")
def complete_week_endpoint(request: CompleteWeekRequest):
    return complete_week(request)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="127.0.0.1", port=8000, reload=True)
