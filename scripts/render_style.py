"""按参考图风格出图：俯视、无标注，只保留 道路/自车/绿色扫掠带/中心高亮。

    python scripts/render_style.py --run outputs/scenarios/<run> --out outputs/scenarios/S1_style.png
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import cv2

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# 参考图配色（BGR）
SWEEP_RGB = (80, 220, 90)      # 半透明绿扫掠带
CORE_RGB = (150, 255, 170)     # 柔和的中心高亮


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True)
    ap.add_argument('--out')
    ap.add_argument('--sweep-extra', type=float, default=0.30, help='扫掠带相对车身外扩 m')
    ap.add_argument('--core-ratio', type=float, default=0.62, help='中心高亮带宽 / 扫掠带宽')
    ap.add_argument('--layers', type=int, default=14, help='边缘柔化层数')
    ap.add_argument('--crop', type=int, default=0, help='1=按内容裁边（非正方形）')
    ap.add_argument('--square', type=int, default=1, help='1=裁剪成正方形（论文用）')
    ap.add_argument('--size', type=int, default=1000, help='正方形边长像素')
    args = ap.parse_args(argv)

    from src.carla_bridge.frame import SceneFrame
    from src.carla_bridge.path_profile import Path as GeomPath

    run = Path(args.run)
    plan = np.load(run/'plan.npz', allow_pickle=True)
    view = np.load(run/'view.npz', allow_pickle=True)
    rgb = view['rgb']
    inv = np.asarray(view['inverse'], float)
    ground = float(view['ground_z'])
    half_w = float(view['half_width'])
    path = np.asarray(plan['curve_world'], float)
    geom = GeomPath.build(path)

    h, w = rgb.shape[:2]

    def px(xy):
        xy = np.asarray(xy, float).reshape(-1, 2)
        hom = np.column_stack([xy, np.full(len(xy), ground), np.ones(len(xy))])
        cam = hom @ inv.T
        if np.any(cam[:, 0] <= 0.05):
            cam[:, 0] = np.maximum(cam[:, 0], 0.05)
        f = w/2.0
        return np.column_stack([w/2 + f*cam[:, 1]/cam[:, 0], h/2 - f*cam[:, 2]/cam[:, 0]])

    def band(half_width_m, steps=0.4):
        left, right = [], []
        for s in np.arange(0.0, geom.s[-1], steps):
            p, yaw = geom.pose(float(s))
            n = np.array([-np.sin(yaw), np.cos(yaw)])
            left.append(p+n*half_width_m)
            right.append(p-n*half_width_m)
        return np.vstack([np.array(left), np.array(right)[::-1]])

    SHIFT = 4
    SCALE = 1 << SHIFT

    def px_sub(points):
        """4-bit 亚像素坐标：避免细线出现整数像素台阶。"""
        return np.rint(px(points)*SCALE).astype(np.int32)

    img = rgb.astype(np.float32)

    # 车身/障碍所在区域不叠加绿色（避免把车染绿）
    vehicle = np.zeros((h, w), np.float32)
    for key in ('ego_corners', 'obstacle_corners'):
        if key not in view:
            continue
        raw = view[key]
        items = raw if key == 'obstacle_corners' else [raw]
        for corners in items:
            pts = px_sub(np.asarray(corners))
            cv2.fillPoly(vehicle, [pts], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
    if vehicle.any():
        # 只挖掉车身内部（腐蚀 6 px），让带子边缘贴着车身，视觉上才是对齐的
        vehicle = cv2.erode(vehicle, np.ones((13, 13), np.uint8))
        vehicle = cv2.GaussianBlur(vehicle, (0, 0), 1.2)
        vehicle = np.clip(vehicle*1.2, 0.0, 1.0)
    keep = (1.0-vehicle)[:, :, None]

    def blend(poly_world, colour, alpha):
        nonlocal img
        pts = px_sub(poly_world)
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [pts], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 3.0)[:, :, None]*keep
        img = img*(1-alpha*mask) + np.array(colour, np.float32)*alpha*mask

    sweep_half = half_w + args.sweep_extra
    # 1) 外层：多层递减 alpha，形成柔和边缘
    for k in range(args.layers):
        t = k/(args.layers-1) if args.layers > 1 else 1.0
        half = sweep_half*(1.0+0.45*(1.0-t))
        blend(band(half), SWEEP_RGB, 0.042)
    # 2) 主体扫掠带（不透明度提高，用户要求绿色更实）
    blend(band(sweep_half), SWEEP_RGB, 0.32)
    blend(band(sweep_half*0.92), SWEEP_RGB, 0.10)
    # 3) 中心柔和高亮
    blend(band(sweep_half*args.core_ratio), CORE_RGB, 0.24)
    blend(band(sweep_half*args.core_ratio*0.6), CORE_RGB, 0.18)

    out = np.clip(img, 0, 255).astype(np.uint8)

    # 4) 中心轨迹线：连续、平滑、细，压在带子上（车身区域不画）
    line_pts = px_sub(path)
    line_mask = np.zeros((h, w), np.uint8)
    cv2.polylines(line_mask, [line_pts], False, 255, 2, cv2.LINE_AA, shift=SHIFT)
    line_mask = cv2.GaussianBlur(line_mask, (0, 0), 0.8).astype(np.float32)/255.0
    line_mask = (line_mask*keep[:, :, 0])[:, :, None]
    out = (out.astype(np.float32)*(1-0.85*line_mask)
           + np.array((240, 252, 246), np.float32)*0.80*line_mask).astype(np.uint8)

    if args.square:
        m = np.zeros((h, w), np.uint8)
        cv2.fillPoly(m, [np.rint(px(band(sweep_half*1.5))).astype(np.int32)], 255)
        ys, xs = np.where(m > 0)
        if len(xs):
            cx, cy = (xs.min()+xs.max())/2.0, (ys.min()+ys.max())/2.0
            need = max(xs.max()-xs.min(), ys.max()-ys.min())*1.35
            side = int(min(max(need, 240), min(h, w)))
            x0 = int(np.clip(cx-side/2, 0, w-side))
            y0 = int(np.clip(cy-side/2, 0, h-side))
            out = out[y0:y0+side, x0:x0+side]
            out = cv2.resize(out, (args.size, args.size), interpolation=cv2.INTER_CUBIC)

    dst = Path(args.out or (run/'style.png'))
    ok, buf = cv2.imencode('.png', out)        # cv2.imwrite 不支持非 ASCII 路径
    if ok:
        dst.write_bytes(buf.tobytes())
    print('saved', dst, out.shape)


if __name__ == '__main__':
    main()
