# Geometry semantics revision

The overall framework now distinguishes three different objects:

1. **Predicted ellipses**: 11 individually visible native ellipse shapes, oriented along the selected skeleton, with amber outlines and light fill.
2. **Convex safe corridor**: overlapping green convex polygons with visible boundaries. The red centerline here remains the selected skeleton route.
3. **Final trajectory**: an actual smooth cubic B-spline, accompanied by 15 representative control points and its control polygon, over the constructed safe corridor. The curve is represented by exact cubic Bezier spans in an editable draw.io stencil, rather than a smoothed skeleton polyline.

The three thumbnails share the same obstacle geometry and endpoints. The numbers of ellipses, polygons and controls are schematic choices, not experiment configuration values. Independent SVG and PNG assets and an enlarged comparison are saved here.

History labels use `\(\mathcal{H}_t\)` at the input, `\(F_{\mathrm{fb},t}\)` after the feedback encoder, and `\(\mathcal{H}_{t+1}\)` on the feedback arrow, matching the diagram's existing `\(Q_t\)` / `\(Q_{t+1}\)` time convention. All 18 mathematical labels use inline `\( ... \)` syntax, with draw.io math rendering enabled.

The pre-revision overall framework is backed up as `总体框架图-语义修订前备份.drawio`. The main file and the `-美化版` / `-几何语义修订版` copies contain the same revised diagram. Run `refine_geometry_semantics.py` to reproduce this revision from that backup; running the older layout-polishing script alone restores the preceding geometry.

`validation.json` records drawing geometry checks. These checks are for this synthetic illustration, not a planner evaluation or a continuous safety proof. The native draw.io full PNG was visually checked, including the LaTeX labels.
