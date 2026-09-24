"""CARLA bridge for the TrajSafe-Diffuser closed-loop demo.

    frame.py            world / local / scene / grid transforms (single source of truth)
    occupancy.py        canonical occupancy from the dataset snapshot or from CARLA
    planner_adapter.py  Engine.generate wrapper + hard acceptance checks
    path_profile.py     arc-length resampling, curvature and speed profile
    controller.py       Pure Pursuit + PID -> Ackermann command
    scenario.py         Town03 scene reproduction (ego, obstacles, cameras)
    demo.py             the closed-loop runner

See CARLA_CLOSED_LOOP_INTEGRATION_PLAN.md for the design this implements.
"""

from .frame import LocalFrame, raw_occupancy_to_canonical, wrap_to_pi
from .rolling_frame import WorldSceneFrame

__all__ = ["LocalFrame", "WorldSceneFrame", "raw_occupancy_to_canonical",
           "wrap_to_pi"]
