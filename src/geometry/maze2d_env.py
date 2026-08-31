"""Pure-Python Maze2D world layouts.

Only the MAZES dict is used by the active pipeline (src/geometry/d4rl_coordinates.py
imports it for the wall/tile geometry).  The full Maze2DEnv double-integrator
simulation and its helper functions were legacy dead code (referenced nowhere
active) and have been removed.
"""
MAZES = {
    'umaze': (
        "#####\n"
        "#GOO#\n"
        "###O#\n"
        "#OOO#\n"
        "#####"
    ),
    'medium': (
        '########\n'
        '#OO##OO#\n'
        '#OO#OOO#\n'
        '##OOO###\n'
        '#OO#OOO#\n'
        '#O#OO#O#\n'
        '#OOO#OG#\n'
        "########"
    ),
    'large': (
        '############\n'
        '#OOOO#OOOOO#\n'
        '#O##O#O#O#O#\n'
        '#OOOOOO#OOO#\n'
        '#O####O###O#\n'
        '#OO#O#OOOOO#\n'
        '##O#O#O#O###\n'
        '#OO#OOO#OGO#\n'
        '############'
    ),
}
