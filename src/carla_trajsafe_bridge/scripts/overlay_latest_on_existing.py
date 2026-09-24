"""Overlay a latest-model trajectory onto the right panel of an existing PNG.

The base figure is intentionally not re-plotted.  Only the new polyline and a
small legend key are composited onto the existing raster image.
"""

from __future__ import annotations

import argparse
import os
import sys

import cv2
import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from src.carla_bridge.demo import build_sample_context, load_yaml  # noqa: E402


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", required=True)
    parser.add_argument("--plan", required=True)
    parser.add_argument("--config", default=os.path.join(
        ROOT, "configs", "carla_demo.yaml"))
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)

    image = cv2.imread(os.path.abspath(args.base), cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(args.base)
    plan = np.load(os.path.abspath(args.plan))
    curve_world = np.asarray(plan["curve_world"], dtype=np.float64)
    ctx = build_sample_context(load_yaml(args.config))
    scene = ctx.frame.to_scene(curve_world)
    grid = (scene + 1.0) * 0.5 * 255.0
    grid[:, 1] = 255.0 - grid[:, 1]

    # Pixel bounds of the right Matplotlib axes in the preserved 1680x882
    # source image.  Scale them if the source is resized without cropping.
    height, width = image.shape[:2]
    sx, sy = width / 1680.0, height / 882.0
    left, right = 900.0 * sx, 1651.0 * sx
    top, bottom = 89.0 * sy, 839.0 * sy
    pixels = np.empty_like(grid)
    pixels[:, 0] = left + grid[:, 0] / 255.0 * (right - left)
    pixels[:, 1] = top + grid[:, 1] / 255.0 * (bottom - top)
    points = np.rint(pixels).astype(np.int32).reshape(-1, 1, 2)

    # Keep the original figure untouched apart from one thin blue curve.
    latest_colour = (220, 90, 40)  # BGR: a restrained royal blue
    cv2.polylines(image, [points], False, latest_colour,
                  max(2, int(round(3 * sx))), cv2.LINE_AA)

    # Add one compact key above the original legend without modifying it.
    x0, y0 = int(round(909 * sx)), int(round(678 * sy))
    x1, y1 = int(round(1215 * sx)), int(round(708 * sy))
    cv2.rectangle(image, (x0, y0), (x1, y1), (205, 205, 205), -1)
    cv2.rectangle(image, (x0, y0), (x1, y1), (120, 120, 120), 1)
    ly = int(round(693 * sy))
    cv2.line(image, (int(round(918 * sx)), ly),
             (int(round(952 * sx)), ly), latest_colour,
             max(2, int(round(3 * sx))), cv2.LINE_AA)
    cv2.putText(image, "latest guided (16-step)",
                (int(round(962 * sx)), int(round(700 * sy))),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48 * sx, (25, 25, 25),
                max(1, int(round(sx))), cv2.LINE_AA)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    if not cv2.imwrite(os.path.abspath(args.out), image):
        raise RuntimeError("failed to write %s" % args.out)
    print(os.path.abspath(args.out))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
