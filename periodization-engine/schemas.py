
from pydantic import BaseModel, Field


class CompleteWeekRequest(BaseModel):
    athlete_id: str
    week_number: int = Field(..., ge=1, le=12)


class AthleteInput(BaseModel):
    athlete_id: str = Field(default="demo-athlete")
    goal: str = Field(default="strength")
    experience: str = Field(default="beginner")
    equipment: str = Field(default="Full Gym")

    days_per_week: int = Field(default=3, ge=2, le=6)
    weeks: int = Field(default=8, ge=4, le=12)
    week_number: int | None = Field(default=None, ge=1, le=12)

    exercises_per_session: int = Field(default=5, ge=3, le=8)
    starting_loads_kg: dict[str, float] = Field(default_factory=dict)

    fatigue: int = Field(default=3, ge=1, le=10)
    performance_declining: bool = False

    simulation_completed_all_sets: bool = True
    simulation_completed_target_reps: bool = True
    simulation_effort: int = Field(default=6, ge=1, le=10)

    excluded_exercises: list[str] = Field(default_factory=list)
    restriction_notes: str = ""


class WorkoutFeedback(BaseModel):
    athlete_id: str
    exercise_name: str
    goal: str = "strength"
    experience: str = "beginner"
    week_number: int = Field(..., ge=1, le=12)

    planned_sets: int = Field(..., ge=1)
    planned_reps: int = Field(..., ge=1)
    completed_sets: int = Field(..., ge=0)
    average_completed_reps: float = Field(..., ge=0)
    effort: int = Field(..., ge=1, le=10)
