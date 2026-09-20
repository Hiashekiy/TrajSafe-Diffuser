"""Diffusion Lens inference service (entry point).

The dashboard now serves the CARLA + 32-control B-spline model on the cleaned
CARLA snapshot, so this file is kept only as the familiar entry point: it runs
backend_carla.py (same HTTP contract: GET /health, POST /generate).

    python backend.py            # http://localhost:8765

The legacy Maze2D service is preserved as backend_maze2d.py.
"""
import os
import runpy
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
TARGET = os.path.join(HERE, "backend_carla.py")
sys.path.insert(0, HERE)
sys.argv = [TARGET] + sys.argv[1:]
runpy.run_path(TARGET, run_name="__main__")

