
from __future__ import annotations

import math
from typing import Any


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(value, high))


def calculate_load(
    starting_load: float | None,
    unit: str = "kg",
    intensity_percent: float | None = None,
) -> float | None:
    """
    Calculate a target load from an explicitly supplied working load.

    intensity_percent is an optional percentage of that working load,
    not a percentage of 1RM. Do not pass dataset intensity here.
    """
    if starting_load is None or starting_load <= 0:
        return None

    if intensity_percent is None:
        return round(starting_load, 2)

    fraction = clamp(intensity_percent, 50, 100) / 100
    return round(starting_load * fraction, 2)


def choose_action(
    completed_all_sets: bool,
    completed_target_reps: bool,
    effort: int,
    fatigue: int,
) -> tuple[str, str]:
    """Return a transparent demonstration progression decision."""
    if fatigue >= 9:
        return "deload", "Very high reported fatigue; reduce training demand."

    if not completed_all_sets:
        return "reduce", "Not all prescribed sets were completed."

    if not completed_target_reps:
        return "maintain", "The target repetitions were not completed."

    if effort >= 9:
        return "maintain", "Very high effort; do not increase the load."

    if fatigue >= 7:
        return "maintain", "Elevated fatigue; hold the current load."

    return "progress", "Targets were completed at manageable effort."


def progress_load(
    current_load: float | None,
    unit: str,
    action: str,
    increment_percent: float = 2.5,
) -> float | None:
    """
    Apply a conservative demo increment to an explicitly supplied load.

    The increment is configurable and should be validated for each lift.
    """
    if current_load is None or current_load <= 0:
        return None

    if action != "progress":
        return round(current_load, 2)

    fraction = clamp(increment_percent, 0.5, 10.0) / 100
    return round(current_load * (1 + fraction), 2)


def make_week_schedule(
    weeks: int,
    fatigue: int = 1,
    performance_declining: bool = False,
) -> list[dict[str, Any]]:
    """
    Create a simple, explainable phase schedule.

    A planned recovery week is scheduled at week 4 when the plan is long
    enough. The final week is not automatically a deload if it is already
    a recovery week or the plan ends earlier.
    """
    schedule: list[dict[str, Any]] = []

    for week in range(1, weeks + 1):
        planned_deload = week % 4 == 0 and week < weeks

        condition_deload = (
            fatigue >= 9
            or (fatigue >= 8 and performance_declining)
        )

        is_deload = planned_deload or condition_deload

        if is_deload:
            phase = "Deload / recovery"
            reason = (
                "Recovery week scheduled by the periodization plan."
                if planned_deload
                else "Fatigue and performance indicators suggest reducing demand."
            )
        elif week <= min(3, weeks):
            phase = "Accumulation"
            reason = "Build training consistency and productive volume."
        elif week == weeks:
            phase = "Realization / consolidation"
            reason = "Consolidate progress without forcing maximal attempts."
        else:
            phase = "Intensification"
            reason = "Prioritize quality work and gradual progression."

        schedule.append({
            "week": week,
            "phase": phase,
            "is_deload": is_deload,
            "phase_note": reason,
            "set_multiplier": 0.6 if is_deload else 1.0,
        })

    return schedule
