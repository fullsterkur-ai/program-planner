
from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from dataset import normalize
from schemas import WorkoutFeedback


ROOT = Path(__file__).resolve().parent

STATE_PATH = ROOT / "data" / "processed" / "athlete_exercise_state.csv"

PROGRAM_STATE_PATH = (
    ROOT / "data" / "processed" / "athlete_program_state.json"
)


def load_adaptive_state(
    athlete_id: str,
    exercise_name: str,
) -> dict[str, Any] | None:
    """Return the latest saved prescription for an athlete and exercise."""
    if not STATE_PATH.exists():
        return None

    try:
        state_df = pd.read_csv(STATE_PATH)
    except (pd.errors.EmptyDataError, pd.errors.ParserError):
        return None

    required = {
        "athlete_id",
        "exercise_name",
        "next_sets",
        "next_reps",
        "updated_at",
        "last_action",
    }

    if not required.issubset(state_df.columns):
        return None

    matches = state_df[
        state_df["athlete_id"].map(normalize).eq(normalize(athlete_id))
        & state_df["exercise_name"].map(normalize).eq(
            normalize(exercise_name)
        )
    ]

    if matches.empty:
        return None

    row = matches.sort_values("updated_at").iloc[-1]

    return {
        "next_sets": int(row["next_sets"]),
        "next_reps": int(row["next_reps"]),
        "updated_at": str(row["updated_at"]),
        "last_action": str(row["last_action"]),
    }


def save_adaptive_state(
    feedback: WorkoutFeedback,
    action: str,
) -> dict[str, Any]:
    """Save the next target prescription after workout feedback."""
    next_sets = feedback.planned_sets
    next_reps = feedback.planned_reps

    if action == "progress":
        next_reps = min(50, feedback.planned_reps + 1)

    elif action == "reduce":
        next_sets = max(1, feedback.planned_sets - 1)

    updated_at = datetime.now(timezone.utc).isoformat()

    new_row = {
        "athlete_id": feedback.athlete_id,
        "exercise_name": feedback.exercise_name,
        "next_sets": next_sets,
        "next_reps": next_reps,
        "last_action": action,
        "updated_at": updated_at,
    }

    STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    if STATE_PATH.exists():
        try:
            state_df = pd.read_csv(STATE_PATH)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            state_df = pd.DataFrame()
    else:
        state_df = pd.DataFrame()

    if not state_df.empty and {
        "athlete_id",
        "exercise_name",
    }.issubset(state_df.columns):
        same_entry = (
            state_df["athlete_id"].map(normalize).eq(
                normalize(feedback.athlete_id)
            )
            & state_df["exercise_name"].map(normalize).eq(
                normalize(feedback.exercise_name)
            )
        )

        state_df = state_df.loc[~same_entry].copy()

    state_df = pd.concat(
        [state_df, pd.DataFrame([new_row])],
        ignore_index=True,
    )

    state_df.to_csv(STATE_PATH, index=False)

    return new_row


def load_program_state(athlete_id: str) -> dict[str, Any]:
    """Load the saved programme state for one athlete."""
    default_state = {
        "athlete_id": athlete_id,
        "weeks": None,
        "current_week": 1,
        "completed_weeks": [],
        "current_loads_kg": {},
        "updated_at": None,
    }

    if not PROGRAM_STATE_PATH.exists():
        return default_state

    try:
        with PROGRAM_STATE_PATH.open("r", encoding="utf-8") as file:
            all_states = json.load(file)
    except (OSError, json.JSONDecodeError):
        return default_state

    if not isinstance(all_states, dict):
        return default_state

    saved = all_states.get(normalize(athlete_id))

    if not isinstance(saved, dict):
        return default_state

    state = {**default_state, **saved}

    if not isinstance(state.get("current_loads_kg"), dict):
        state["current_loads_kg"] = {}

    if not isinstance(state.get("completed_weeks"), list):
        state["completed_weeks"] = []

    return state


def save_program_state(
    athlete_id: str,
    state: dict[str, Any],
) -> None:
    """Persist programme state without overwriting other athletes."""
    PROGRAM_STATE_PATH.parent.mkdir(parents=True, exist_ok=True)

    all_states = {}

    if PROGRAM_STATE_PATH.exists():
        try:
            with PROGRAM_STATE_PATH.open("r", encoding="utf-8") as file:
                loaded = json.load(file)

            if isinstance(loaded, dict):
                all_states = loaded

        except (OSError, json.JSONDecodeError):
            all_states = {}

    state["athlete_id"] = athlete_id
    state["updated_at"] = datetime.now(timezone.utc).isoformat()

    all_states[normalize(athlete_id)] = state

    temp_path = PROGRAM_STATE_PATH.with_suffix(".tmp")

    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(all_states, file, indent=2, ensure_ascii=False)

    temp_path.replace(PROGRAM_STATE_PATH)
