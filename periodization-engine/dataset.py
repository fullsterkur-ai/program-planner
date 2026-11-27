
from __future__ import annotations

import ast
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd

from schemas import AthleteInput


ROOT = Path(__file__).resolve().parent

DATA_PATH = (
    ROOT / "data" / "raw" / "programs_detailed_boostcamp_kaggle.csv"
)

MODEL_PATH = ROOT / "models" / "prescription_model.joblib"

ML_BUNDLE = None


GOAL_TERMS = {
    "strength": [
        "strength",
        "powerlifting",
        "powerbuilding",
        "athletics",
    ],
    "hypertrophy": [
        "bodybuilding",
        "muscle",
        "sculpting",
        "hypertrophy",
        "powerbuilding",
    ],
    "general_fitness": [
        "fitness",
        "athletics",
        "general",
        "muscle",
    ],
}

LEVEL_ALIASES = {
    "beginner": ["beginner", "novice"],
    "novice": ["novice", "beginner"],
    "intermediate": ["intermediate"],
    "advanced": ["advanced"],
}

EQUIPMENT_ALIASES = {
    "full gym": ["full gym", "gym"],
    "garage gym": ["garage gym", "garage"],
    "at home": ["at home", "home"],
    "dumbbell only": ["dumbbell only", "dumbbell"],
}


def safe_text(value: Any) -> str:
    if value is None:
        return ""

    try:
        if pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass

    if isinstance(value, (list, tuple, set)):
        return " ".join(str(x) for x in value)

    value = str(value).strip()

    if value.startswith("[") and value.endswith("]"):
        try:
            parsed = ast.literal_eval(value)

            if isinstance(parsed, (list, tuple, set)):
                return " ".join(str(x) for x in parsed)

        except (ValueError, SyntaxError):
            pass

    return value


def normalize(value: Any) -> str:
    return re.sub(r"\s+", " ", safe_text(value).lower()).strip()


def find_column(
    df: pd.DataFrame,
    candidates: list[str],
) -> str | None:
    lookup = {
        str(col).strip().lower(): col
        for col in df.columns
    }

    return next(
        (
            lookup[c.lower()]
            for c in candidates
            if c.lower() in lookup
        ),
        None,
    )


def load_dataset() -> pd.DataFrame:
    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found at {DATA_PATH}. "
            "Check that the expected CSV exists in data/raw/."
        )

    df = pd.read_csv(DATA_PATH, low_memory=False)
    df.columns = [str(c).strip() for c in df.columns]

    required = ["exercise_name", "sets", "reps"]
    missing = [c for c in required if c not in df.columns]

    if missing:
        raise ValueError(
            f"CSV is missing expected columns: {missing}. "
            f"Found columns: {list(df.columns)}"
        )

    df = df.drop_duplicates().copy()

    df["exercise_name"] = (
        df["exercise_name"].map(safe_text).str.strip()
    )

    df = df[df["exercise_name"].ne("")].copy()

    for col in ["sets", "reps", "intensity", "week", "day"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Negative reps in this dataset encode duration in seconds.
    df["duration_seconds"] = np.where(
        df["reps"] < 0,
        df["reps"].abs(),
        np.nan,
    )

    df["rep_count"] = df["reps"].where(
        df["reps"] > 0,
        np.nan,
    )

    return df


def list_match(value: Any, terms: list[str]) -> bool:
    value_norm = normalize(value)
    return any(term in value_norm for term in terms)


def candidate_rows(
    df: pd.DataFrame,
    athlete: AthleteInput,
) -> pd.DataFrame:
    candidates = df.copy()

    goal_col = find_column(candidates, ["goal", "goals"])

    level_col = find_column(
        candidates,
        ["level", "experience", "experience_level"],
    )

    equipment_col = find_column(
        candidates,
        ["equipment", "equipment_type"],
    )

    if goal_col:
        terms = GOAL_TERMS.get(
            normalize(athlete.goal),
            [normalize(athlete.goal)],
        )

        mask = candidates[goal_col].map(
            lambda x: list_match(x, terms)
        )

        if mask.any():
            candidates = candidates[mask].copy()

    if level_col:
        terms = LEVEL_ALIASES.get(
            normalize(athlete.experience),
            [normalize(athlete.experience)],
        )

        mask = candidates[level_col].map(
            lambda x: list_match(x, terms)
        )

        if mask.any():
            candidates = candidates[mask].copy()

    if equipment_col:
        terms = EQUIPMENT_ALIASES.get(
            normalize(athlete.equipment),
            [normalize(athlete.equipment)],
        )

        mask = candidates[equipment_col].map(
            lambda x: list_match(x, terms)
        )

        if mask.any():
            candidates = candidates[mask].copy()

    return candidates


def summarize_exercises(
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    rows = []

    for name, group in candidates.groupby(
        "exercise_name",
        dropna=True,
    ):
        set_values = pd.to_numeric(
            group["sets"],
            errors="coerce",
        )

        set_values = set_values[set_values.between(1, 10)]

        rep_values = pd.to_numeric(
            group["rep_count"],
            errors="coerce",
        )

        rep_values = rep_values[rep_values.between(1, 50)]

        duration_values = pd.to_numeric(
            group["duration_seconds"],
            errors="coerce",
        )

        duration_values = duration_values[
            duration_values.between(1, 3600)
        ]

        if "intensity" in group.columns:
            intensity_values = pd.to_numeric(
                group["intensity"],
                errors="coerce",
            )
        else:
            intensity_values = pd.Series(dtype=float)

        if not rep_values.empty:
            prescription_type = "reps"
            target_reps = int(
                round(float(rep_values.median()))
            )
            duration_seconds = None

        elif not duration_values.empty:
            prescription_type = "time"
            target_reps = None
            duration_seconds = int(
                round(float(duration_values.median()))
            )

        else:
            continue

        rows.append({
            "exercise_name": str(name),
            "sets": (
                int(round(float(set_values.median())))
                if not set_values.empty
                else 3
            ),
            "prescription_type": prescription_type,
            "reps": target_reps,
            "duration_seconds": duration_seconds,
            "dataset_intensity_0_10": (
                round(float(intensity_values.median()), 1)
                if not intensity_values.dropna().empty
                else None
            ),
            "record_count": int(len(group)),
        })

    result = pd.DataFrame(rows)

    if result.empty:
        return result

    return result.sort_values(
        ["record_count", "exercise_name"],
        ascending=[False, True],
    )


def normalize_ml_goal(goal: str) -> str:
    value = normalize(goal)

    if any(
        x in value
        for x in [
            "strength",
            "powerlifting",
            "powerbuilding",
            "athletics",
        ]
    ):
        return "strength"

    if any(
        x in value
        for x in [
            "hypertrophy",
            "bodybuilding",
            "muscle",
            "sculpting",
        ]
    ):
        return "hypertrophy"

    return "general_fitness"


def normalize_ml_experience(experience: str) -> str:
    value = normalize(experience)

    if value in ("beginner", "novice"):
        return "beginner"

    if value == "intermediate":
        return "intermediate"

    if value == "advanced":
        return "advanced"

    return "unknown"


def normalize_ml_equipment(equipment: str) -> str:
    value = normalize(equipment)

    if "dumbbell" in value:
        return "dumbbell only"

    if "garage" in value:
        return "garage gym"

    if "home" in value:
        return "at home"

    if "gym" in value:
        return "full gym"

    return "unknown"


def load_ml_bundle():
    global ML_BUNDLE

    if ML_BUNDLE is None and MODEL_PATH.exists():
        ML_BUNDLE = joblib.load(MODEL_PATH)

    return ML_BUNDLE


def predict_prescription(
    athlete: AthleteInput,
    exercise: dict,
    week_no: int,
    day_no: int,
):
    bundle = load_ml_bundle()

    # Keep the existing fallback if no model bundle is available.
    if bundle is None:
        return None

    features = pd.DataFrame([{
        "goal": normalize_ml_goal(athlete.goal),
        "experience": normalize_ml_experience(athlete.experience),
        "equipment": normalize_ml_equipment(athlete.equipment),
        "exercise_name": exercise["exercise_name"],
        "program_length": athlete.weeks,
        "time_per_workout": np.nan,
        "week": week_no,
        "day": day_no,
    }])

    predicted_sets = bundle["sets_model"].predict(features)[0]

    predicted_sets = int(
        np.clip(round(float(predicted_sets)), 1, 10)
    )

    predicted_reps = None

    if exercise["prescription_type"] == "reps":
        predicted_reps = bundle["reps_model"].predict(features)[0]

        predicted_reps = int(
            np.clip(round(float(predicted_reps)), 1, 50)
        )

    return {
        "sets": predicted_sets,
        "reps": predicted_reps,
        "model_used": True,
    }
