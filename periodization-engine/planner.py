
from __future__ import annotations

import math
from typing import Any

import pandas as pd
from fastapi import HTTPException

from schemas import AthleteInput
from dataset import (
    candidate_rows,
    load_dataset,
    load_ml_bundle,
    normalize,
    summarize_exercises,
)
from state_store import (
    load_adaptive_state,
    load_program_state,
    save_program_state,
)
from periodization import choose_action, make_week_schedule



def make_plan(athlete: AthleteInput) -> dict[str, Any]:
    try:
        df = load_dataset()
    except (FileNotFoundError, ValueError, pd.errors.ParserError) as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc

    # Load the athlete's saved programme state.
    state = load_program_state(athlete.athlete_id)

    # Reset programme-week tracking if the requested programme duration changes.
    if state.get("weeks") != athlete.weeks:
        state["weeks"] = athlete.weeks
        state["current_week"] = 1
        state["completed_weeks"] = []

    saved_week = int(state.get("current_week", 1))
    week_no = athlete.week_number if athlete.week_number is not None else saved_week

    if week_no > athlete.weeks:
        raise HTTPException(
            status_code=422,
            detail=f"Week {week_no} exceeds the {athlete.weeks}-week programme.",
        )

    # Restore saved loads, then apply any manual starting-load overrides.
    current_loads = {
        normalize(name): float(load)
        for name, load in state.get("current_loads_kg", {}).items()
        if load is not None
        and math.isfinite(float(load))
        and float(load) > 0
    }

    for name, load in athlete.starting_loads_kg.items():
        if load is not None and math.isfinite(float(load)) and float(load) > 0:
            current_loads[normalize(name)] = float(load)

    state["current_loads_kg"] = current_loads

    catalogue = summarize_exercises(candidate_rows(df, athlete))

    if catalogue.empty:
        catalogue = summarize_exercises(df)
        selection_note = (
            "The requested metadata filters returned no usable records. "
            "The overall dataset catalogue was used instead."
        )
    else:
        selection_note = (
            "Exercise candidates were filtered using available program metadata."
        )

    if catalogue.empty:
        raise HTTPException(
            status_code=422,
            detail="No usable exercises found in the program dataset.",
        )

    catalogue = catalogue.reset_index(drop=True)

    excluded = {normalize(name) for name in athlete.excluded_exercises}

    if excluded:
        catalogue = catalogue[
            ~catalogue["exercise_name"].map(normalize).isin(excluded)
        ].reset_index(drop=True)

    if catalogue.empty:
        raise HTTPException(
            status_code=422,
            detail="All available exercises were excluded. Change the exclusions.",
        )

    # Keep the existing demo action visible, but do not use it to
    # automatically increase loads just because a plan was generated.
    simulated_action, simulated_reason = choose_action(
        completed_all_sets=athlete.simulation_completed_all_sets,
        completed_target_reps=athlete.simulation_completed_target_reps,
        effort=athlete.simulation_effort,
        fatigue=athlete.fatigue,
    )

    schedule = make_week_schedule(
        weeks=athlete.weeks,
        fatigue=athlete.fatigue,
        performance_declining=athlete.performance_declining,
    )

    week_info = schedule[week_no - 1]
    is_deload = week_info["is_deload"]
    set_multiplier = week_info["set_multiplier"]

    sessions = []

    for day_no in range(1, athlete.days_per_week + 1):
        day_catalogue = catalogue.iloc[
            (day_no - 1)::athlete.days_per_week
        ].head(athlete.exercises_per_session)

        exercises = []

        for item in day_catalogue.to_dict(orient="records"):
            exercise_name = item["exercise_name"]
            exercise_key = normalize(exercise_name)

            # Use the latest saved sets/reps when feedback exists.
            adaptive = load_adaptive_state(
                athlete.athlete_id,
                exercise_name,
            )

            base_sets = (
                adaptive["next_sets"]
                if adaptive is not None
                else int(item["sets"])
            )

            base_reps = (
                adaptive["next_reps"]
                if adaptive is not None
                else item["reps"]
            )

            final_sets = max(
                1,
                int(math.floor(base_sets * set_multiplier)),
            )

            last_action = (
                adaptive["last_action"]
                if adaptive is not None
                else "awaiting_feedback"
            )

            entry = {
                "exercise_name": exercise_name,
                "sets": final_sets,
                "prescription_type": item["prescription_type"],
                "dataset_intensity_0_10": item["dataset_intensity_0_10"],
                "prescription_source": (
                    "saved_adaptive_feedback"
                    if adaptive is not None
                    else "dataset_median_plus_rule_based_periodization"
                ),
                "phase": week_info["phase"],
                "is_deload": is_deload,
                "progression_action": (
                        "maintain_during_deload"
                        if is_deload
                        else "awaiting_workout_feedback"
                    ),
                "last_recorded_action": last_action,
                "progression_reason": (
                        "Deload week: avoid load increases."
                        if is_deload
                        else (
                            "Previous feedback is available, but this week's workout "
                            "must still be recorded before progression is applied."
                        )
                    ),
            }

            if item["prescription_type"] == "time":
                entry["duration_seconds"] = item["duration_seconds"]
            else:
                entry["reps"] = int(base_reps)

            current_load = current_loads.get(exercise_key)

            if current_load is not None:
                entry["load_kg"] = round(current_load, 2)
                entry["load_source"] = "saved_or_athlete_supplied_working_load"

                # A proposed load increase is not applied until actual
                # workout feedback and week completion are recorded.
                entry["next_exposure_load_kg"] = round(current_load, 2)
                entry["load_progression_status"] = "pending_workout_feedback"
            else:
                entry["load_kg"] = None
                entry["load_source"] = "not_provided"
                entry["next_exposure_load_kg"] = None
                entry["load_progression_status"] = "starting_load_required"

            exercises.append(entry)

        sessions.append({
            "day": day_no,
            "exercises": exercises,
        })

    # Persist known loads and programme duration. Generating a plan
    # deliberately does NOT mark the week complete or advance the week.
    # Save the exercises expected for this week's feedback.
    planned_by_week = state.setdefault("planned_exercises_by_week", {})

    planned_by_week[str(week_no)] = sorted({
        normalize(exercise["exercise_name"])
        for session in sessions
        for exercise in session["exercises"]
    })

    # Persist known loads and programme state.
    # Generating a plan does not complete or advance the week.
    state["current_loads_kg"] = current_loads
    save_program_state(athlete.athlete_id, state)

    return {
        "athlete": athlete.model_dump(),
        "programme": {
            "total_weeks": athlete.weeks,
            "requested_week": week_no,
            "saved_current_week": state["current_week"],
            "completed_weeks": state["completed_weeks"],
            "week_completed": week_no in state["completed_weeks"],
            "advance_requires_completion_confirmation": True,
        },
        "engine_type": (
            "Rule-based periodization with dataset-informed prescriptions "
            "and saved athlete state"
        ),
        "ml_model_loaded": load_ml_bundle() is not None,
        "dataset_rows_after_exact_deduplication": int(len(df)),
        "candidate_exercise_count": int(len(catalogue)),
        "selection_note": selection_note,
        "simulated_condition": {
            "fatigue": athlete.fatigue,
            "performance_declining": athlete.performance_declining,
            "completed_all_sets": athlete.simulation_completed_all_sets,
            "completed_target_reps": athlete.simulation_completed_target_reps,
            "effort": athlete.simulation_effort,
            "demo_recommended_action": simulated_action,
            "demo_reason": simulated_reason,
            "note": (
                "These simulation inputs are for demonstration only. "
                "They do not confirm that the workout was actually completed."
            ),
        },
        "restrictions": {
            "excluded_exercises": athlete.excluded_exercises,
            "notes": athlete.restriction_notes,
            "warning": (
                "Exercise exclusions are user-selected demo filters, not "
                "medical clearance or a determination that other exercises "
                "are safe for an injury."
            ),
        },
        "limitations": [
            "Research/demo prototype, not clinically validated.",
            "Dataset intensity remains on its original 0-10 scale; it is not assumed to be RPE or a percentage of 1RM.",
            "The Random Forest does not control this plan.",
            "Load progression is not applied automatically; actual workout feedback and completion confirmation are required.",
            "Exercises without a supplied or saved working load retain a null load.",
            "Exercise selection does not yet optimize muscle-group balance.",
        ],
        "weeks": [{
            "week": week_no,
            "phase": week_info["phase"],
            "is_deload": is_deload,
            "phase_note": week_info["phase_note"],
            "sessions": sessions,
        }],
    }

