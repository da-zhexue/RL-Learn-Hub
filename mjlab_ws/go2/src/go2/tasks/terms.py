"""Names of all reward terms, in the same order as mjlab's reward config."""


TERM_NAMES: tuple[str, ...] = (
    "progress",
    "track_lin_vel",
    "climb",
    "level_bonus",
    "feet_clearance",
    "feet_air_time",
    "foot_swing_height",
    "base_height",
    "lateral",
    "lateral_vel",
    "orientation",
    "yaw",
    "yaw_rate",
    "action_rate",
    "torques",
    "collision",
    "success",
    "fall",
)
