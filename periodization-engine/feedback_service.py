
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from fastapi import HTTPException

from schemas import WorkoutFeedback, CompleteWeekRequest
from dataset import normalize
from periodization import progress_load
from state_store import (
    load_program_state,
    save_program_state,
    save_adaptive_state,
)


ROOT = Path(__file__).resolve().parent

FEEDBACK_PATH = ROOT / "data" / "processed" / "workout_feedback.csv"


def record_workout_feedback(
    feedback: WorkoutFeedback,
) -> dict[str, Any]:
    """Validate and save workout feedback and the next prescription."""

    if feedback.completed_sets < feedback.planned_sets:
        action = "reduce"
        reason = "Not all planned sets were completed."

    elif feedback.average_completed_reps < feedback.planned_reps:
        action = "repeat"
        reason = "The planned repetition target was not achieved."

    elif feedback.effort >= 9:
        action = "repeat"
        reason = "The reported effort was very high."

    else:
        action = "progress"
        reason = (
            "The planned work was completed without "
            "a very high effort rating."
        )

    program_state = load_program_state(feedback.athlete_id)

    if program_state.get("weeks") is None:
        raise HTTPException(
            status_code=409,
            detail="Generate a programme before recording workout feedback.",
        )

    current_week = int(program_state.get("current_week", 1))
    week_number = feedback.week_number or current_week

    if week_number != current_week:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The athlete's current week is {current_week}. "
                f"Feedback for Week {week_number} cannot be recorded."
            ),
        )

    planned_by_week = program_state.get("planned_exercises_by_week", {})
    planned_exercises = planned_by_week.get(str(week_number))

    if not planned_exercises:
        raise HTTPException(
            status_code=409,
            detail=(
                "The planned exercises for this week have not been saved. "
                "Generate this week's plan again before submitting feedback."
            ),
        )

    exercise_key = normalize(feedback.exercise_name)

    if exercise_key not in planned_exercises:
        raise HTTPException(
            status_code=422,
            detail=(
                "This exercise is not in the saved plan for "
                f"Week {week_number}."
            ),
        )

    saved_feedback = feedback.model_copy(
        update={"week_number": week_number}
    )

    FEEDBACK_PATH.parent.mkdir(parents=True, exist_ok=True)

    if FEEDBACK_PATH.exists():
        try:
            feedback_df = pd.read_csv(FEEDBACK_PATH)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            raise HTTPException(
                status_code=500,
                detail=(
                    "The existing workout feedback CSV could not be read. "
                    "Check the file before submitting more feedback."
                ),
            )
    else:
        feedback_df = pd.DataFrame()

    for column in [
        "athlete_id",
        "exercise_name",
        "week_number",
    ]:
        if column not in feedback_df.columns:
            feedback_df[column] = pd.NA

    if not feedback_df.empty:
        same_athlete = (
            feedback_df["athlete_id"]
            .fillna("")
            .astype(str)
            .map(normalize)
            .eq(normalize(saved_feedback.athlete_id))
        )

        same_exercise = (
            feedback_df["exercise_name"]
            .fillna("")
            .astype(str)
            .map(normalize)
            .eq(exercise_key)
        )

        recorded_weeks = pd.to_numeric(
            feedback_df["week_number"],
            errors="coerce",
        )

        duplicate = (
            same_athlete
            & same_exercise
            & recorded_weeks.eq(week_number)
        )

        if duplicate.any():
            raise HTTPException(
                status_code=409,
                detail=(
                    f"Feedback for {saved_feedback.exercise_name} "
                    f"has already been recorded for Week {week_number}."
                ),
            )

    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        **saved_feedback.model_dump(),
        "recommended_action": action,
        "reason": reason,
    }

    feedback_df = pd.concat(
        [feedback_df, pd.DataFrame([record])],
        ignore_index=True,
    )

    feedback_df.to_csv(FEEDBACK_PATH, index=False)

    next_prescription = save_adaptive_state(
        saved_feedback,
        action,
    )

    pending_by_week = program_state.setdefault(
        "pending_load_actions",
        {},
    )

    pending_actions = pending_by_week.setdefault(
        str(week_number),
        {},
    )

    pending_actions[exercise_key] = action

    save_program_state(
        saved_feedback.athlete_id,
        program_state,
    )

    return {
        "athlete_id": saved_feedback.athlete_id,
        "exercise_name": saved_feedback.exercise_name,
        "week_number": week_number,
        "recommended_action": action,
        "reason": reason,
        "feedback_saved": True,
        "adaptive_state_saved": True,
        "load_change": "pending_week_completion",
        "next_prescription": {
            "sets": next_prescription["next_sets"],
            "reps": next_prescription["next_reps"],
        },
        "message": (
            "Feedback recorded. The next sets/reps prescription "
            "has been saved. Any working-load progression will be "
            "applied when the week is confirmed complete. "
            "Recommendations use explicit rules, not the "
            "Random Forest model."
        ),
    }


def complete_week(
    request: CompleteWeekRequest,
) -> dict[str, Any]:
    """Validate completion and apply pending load changes."""

    state = load_program_state(request.athlete_id)
    total_weeks = state.get("weeks")

    if total_weeks is None:
        raise HTTPException(
            status_code=409,
            detail="Generate a programme before completing a week.",
        )

    current_week = int(state.get("current_week", 1))
    week_number = request.week_number

    if week_number != current_week:
        raise HTTPException(
            status_code=409,
            detail=(
                f"The saved current week is {current_week}. "
                f"Cannot complete Week {week_number}."
            ),
        )

    completed_weeks = state.get("completed_weeks", [])

    if week_number in completed_weeks:
        raise HTTPException(
            status_code=409,
            detail=f"Week {week_number} is already completed.",
        )

    planned_by_week = state.get("planned_exercises_by_week", {})
    planned_exercises = planned_by_week.get(str(week_number), [])

    if not planned_exercises:
        raise HTTPException(
            status_code=409,
            detail=(
                "No saved exercise list exists for this week. "
                "Generate the current week's plan again before "
                "recording completion."
            ),
        )

    if FEEDBACK_PATH.exists():
        try:
            feedback_df = pd.read_csv(FEEDBACK_PATH)
        except (pd.errors.EmptyDataError, pd.errors.ParserError):
            raise HTTPException(
                status_code=500,
                detail="The workout feedback CSV could not be read.",
            )
    else:
        feedback_df = pd.DataFrame()

    required_columns = {
        "athlete_id",
        "exercise_name",
        "week_number",
    }

    if not required_columns.issubset(feedback_df.columns):
        raise HTTPException(
            status_code=409,
            detail=(
                "The feedback file does not contain all required "
                "columns. Record workout feedback before completing "
                "this week."
            ),
        )

    if feedback_df.empty:
        week_feedback = feedback_df.copy()
    else:
        same_athlete = (
            feedback_df["athlete_id"]
            .fillna("")
            .astype(str)
            .map(normalize)
            .eq(normalize(request.athlete_id))
        )

        recorded_weeks = pd.to_numeric(
            feedback_df["week_number"],
            errors="coerce",
        )

        week_feedback = feedback_df[
            same_athlete & recorded_weeks.eq(week_number)
        ]

    recorded_exercises = set(
        week_feedback["exercise_name"]
        .fillna("")
        .astype(str)
        .map(normalize)
    )

    planned_set = {
        normalize(name)
        for name in planned_exercises
    }

    missing_exercises = sorted(
        planned_set - recorded_exercises
    )

    if missing_exercises:
        raise HTTPException(
            status_code=409,
            detail={
                "message": (
                    f"Week {week_number} cannot be completed yet. "
                    "Record feedback for every planned exercise."
                ),
                "athlete_id": request.athlete_id,
                "week_number": week_number,
                "missing_feedback_count": len(missing_exercises),
                "missing_exercises": missing_exercises,
            },
        )

    current_loads = state.get("current_loads_kg", {})
    if not isinstance(current_loads, dict):
        current_loads = {}

    pending_by_week = state.get("pending_load_actions", {})
    pending_actions = pending_by_week.get(str(week_number), {})

    applied_load_changes = []
    skipped_load_changes = []

    load_keys = {
        normalize(name): name
        for name in current_loads
    }

    for exercise_key, action in pending_actions.items():
        original_key = load_keys.get(normalize(exercise_key))

        if original_key is None or current_loads[original_key] is None:
            skipped_load_changes.append({
                "exercise_name": exercise_key,
                "action": action,
                "reason": "No working load has been supplied or saved.",
            })
            continue

        try:
            old_load = float(current_loads[original_key])
        except (TypeError, ValueError):
            skipped_load_changes.append({
                "exercise_name": exercise_key,
                "action": action,
                "reason": "The saved working load is not numeric.",
            })
            continue

        if action not in {"progress", "reduce", "repeat"}:
            skipped_load_changes.append({
                "exercise_name": exercise_key,
                "action": action,
                "reason": "Unknown load-progression action.",
            })
            continue

        new_load = progress_load(old_load, "kg", action)
        current_loads[original_key] = new_load

        applied_load_changes.append({
            "exercise_name": original_key,
            "action": action,
            "previous_load_kg": old_load,
            "next_load_kg": new_load,
        })

    completed_weeks.append(week_number)
    state["completed_weeks"] = sorted(set(completed_weeks))
    state["current_loads_kg"] = current_loads

    pending_by_week.pop(str(week_number), None)
    state["pending_load_actions"] = pending_by_week

    if week_number < int(total_weeks):
        state["current_week"] = week_number + 1
        message = (
            f"Week {week_number} completed. "
            f"Ready for Week {week_number + 1}."
        )
    else:
        state["current_week"] = week_number
        message = "The final programme week is completed."

    save_program_state(request.athlete_id, state)

    return {
        "athlete_id": request.athlete_id,
        "completed_week": week_number,
        "current_week": state["current_week"],
        "total_weeks": total_weeks,
        "completed_weeks": state["completed_weeks"],
        "programme_finished": week_number >= int(total_weeks),
        "applied_load_changes": applied_load_changes,
        "skipped_load_changes": skipped_load_changes,
        "message": message,
    }
