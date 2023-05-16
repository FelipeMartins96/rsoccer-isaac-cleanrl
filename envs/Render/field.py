from dataclasses import dataclass

@dataclass()
class Field:
    length: float = 1.5
    width: float = 1.3
    penalty_length: float = 0.15
    penalty_width: float = 0.7
    goal_width: float = 0.4
    goal_depth: float = 0.1
    ball_radius: float = 0.0215
    rbt_wheel0_angle: float = 90
    rbt_wheel1_angle: float = 270
    rbt_radius: float  = 0.0375