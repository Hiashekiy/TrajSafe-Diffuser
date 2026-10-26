# Continuous trajectory safety and ALM correction

`alm_continuous_safety.svg` is editable vector artwork with Times New Roman text. The PDF embeds TrueType fonts, and the PNG is exported at 300 dpi. Run `python draw_alm.py` to regenerate all three.

This is one unified figure, without subfigure labels or a title banner. Blue is the original cubic B-spline; red dashed is the corrected spline; gray shows the original control polygon; red hollow circles show corrected spline control points. Orange convex regions form the assigned illustrative corridor.

The example is synthetic, not an output from the project planner or an experiment. Nine spline control points, four convex regions and the pictured knot vector are demonstration choices. The responsibility boundaries and knots are combined to form nine exact cubic Bezier subsegments. Segment control points are computed from the cubic spline and its endpoint derivatives, and the conversion is checked numerically.

ALM is actually solved for this synthetic example: minimize squared displacement from the original spline control points, subject to every extracted Bezier control point satisfying all halfspaces of its assigned convex region. Start and goal are fixed. The local inset shows the actual selected subsegment and its assigned region. Convexity gives containment of the entire cubic segment when its four control points are inside that region. This statement concerns the illustrated corridor only; it does not assert that the corridor was validated against a real obstacle map.

`geometry_and_validation.json` records the demonstration geometry, ALM iterations and residual checks. No performance claims or project configuration values are implied.
