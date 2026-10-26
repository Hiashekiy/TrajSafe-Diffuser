"""批量出图：调 dashboard 接口跑场景，裁剪成 700x700 正方形（论文风格）。

    python scripts/batch_dashboard_figs.py --port 8857 --out outputs/scenarios/pick --n 40
"""
from __future__ import annotations
import argparse
import base64
import json
import re
import sys
import urllib.request
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
SPAN, RES = 160.0, 256
GOOD = [12, 15, 26, 44, 45, 47, 50, 60, 61, 98, 99, 101, 108, 118]


def post(port, path, body=None, timeout=900):
    req = urllib.request.Request('http://127.0.0.1:%d%s' % (port, path),
                                 data=json.dumps(body or {}).encode(),
                                 headers={'Content-Type': 'application/json'})
    return json.load(urllib.request.urlopen(req, timeout=timeout))


def uv_of(reg, x, y):
    return (((y-(reg['cy']-SPAN/2))/(SPAN/RES))/(RES-1),
            ((x-(reg['cx']-SPAN/2))/(SPAN/RES))/(RES-1))


def crop_square(img, out_px=700, pad=1.42):
    """按绿色扫掠带的包围盒裁正方形。"""
    b, g, r = img[:, :, 0].astype(int), img[:, :, 1].astype(int), img[:, :, 2].astype(int)
    band = (g > 120) & (g-b > 40) & (g-r > 40)
    ys, xs = np.where(band)
    h, w = img.shape[:2]
    if len(xs) < 50:
        return None
    cx, cy = (xs.min()+xs.max())/2.0, (ys.min()+ys.max())/2.0
    need = max(xs.max()-xs.min(), ys.max()-ys.min())*pad
    side = int(min(max(need, 320), min(h, w)))
    x0 = int(np.clip(cx-side/2, 0, w-side))
    y0 = int(np.clip(cy-side/2, 0, h-side))
    out = img[y0:y0+side, x0:x0+side]
    return cv2.resize(out, (out_px, out_px), interpolation=cv2.INTER_CUBIC)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--port', type=int, default=8857)
    ap.add_argument('--out', default='outputs/scenarios/pick')
    ap.add_argument('--n', type=int, default=40)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    import carla
    c = carla.Client('127.0.0.1', 2000)
    c.set_timeout(30.0)
    cmap = c.get_world().get_map()
    sps = cmap.get_spawn_points()
    key2idx = {(round(sps[i].location.x, 1), round(sps[i].location.y, 1)): i for i in GOOD}

    def lane_point(x, y, steps):
        wp = cmap.get_waypoint(carla.Location(x=float(x), y=float(y), z=1.0),
                               project_to_road=True)
        p = wp
        for _ in range(steps):
            nxt = [q for q in p.next(2.0) if q.lane_id == p.lane_id]
            if not nxt:
                break
            p = nxt[0]
        return p, wp

    saved, tried = 0, 0
    seen = set()
    while saved < args.n and tried < args.n*4:
        tried += 1
        reg = post(args.port, '/api/map')['region']
        idx = key2idx.get((round(reg['cx'], 1), round(reg['cy'], 1)))
        if idx is None:
            continue
        mode = tried % 3                     # 0=无障碍 1=本车道有车 2=邻车道有车
        dist = 50 if tried % 2 else 65
        ex, ey = reg['cx'], reg['cy']          # 区域中心 = 该 spawn 点，直接用世界坐标
        try:
            pg, wp = lane_point(ex, ey, int(dist/2))
            po, _ = lane_point(ex, ey, int((dist*0.45)/2))
        except Exception:
            continue
        obs = []
        if mode:
            if mode == 2:
                left = po.get_left_lane()
                tgt = left if (left is not None and left.lane_type.name == 'Driving'
                               and np.sign(left.lane_id) == np.sign(po.lane_id)) else po
            else:
                tgt = po
            obs = [[tgt.transform.location.x, tgt.transform.location.y, 'car']]
        post(args.port, '/api/set_pose', dict(
            ego=[ex, ey],
            goal=[pg.transform.location.x, pg.transform.location.y],
            obstacles=obs))
        r = post(args.port, '/api/plan', dict(seed=43 + tried))
        msg = (r.get('msg') or '')
        if not r.get('ok') or '不足' in msg:
            continue
        mL = re.search(r'轨迹 ([0-9.]+) m', msg)
        mG = re.search(r'终点 ([0-9.]+) m', msg)
        if not (mL and mG):
            continue
        L, G = float(mL.group(1)), float(mG.group(1))
        if L < 40 or abs(L-G) > 15:          # 轨迹长度必须接近目标距离，剔掉退化结果
            continue
        m = re.search(r'最小间隙 ([0-9.]+)', msg)
        if m and float(m.group(1)) < 0.35:
            continue
        img = cv2.imdecode(np.frombuffer(base64.b64decode(r['view']), np.uint8),
                           cv2.IMREAD_COLOR)
        sq = crop_square(img)
        if sq is None:
            continue
        tag = 's%03d_%s_%dm' % (idx, ('n', 'in', 'adj')[mode], dist)
        f = out/('%03d_%s.png' % (saved+1, tag))
        out.joinpath(f.name).write_bytes(cv2.imencode('.png', sq)[1].tobytes())
        saved += 1
        print('saved %s  %s' % (f.name, msg[:80]), flush=True)

    files = sorted(out.glob('*.png'))
    if files:
        cols = 6
        rows = (len(files)+cols-1)//cols
        cell = 260
        sheet = np.full((rows*cell, cols*cell, 3), 30, np.uint8)
        for k, f in enumerate(files):
            im = cv2.imread(str(f))
            if im is None:
                continue
            im = cv2.resize(im, (cell-8, cell-8))
            r0, c0 = (k//cols)*cell, (k % cols)*cell
            sheet[r0+4:r0+cell-4, c0+4:c0+cell-4] = im
        out.joinpath('contact_sheet.png').write_bytes(
            cv2.imencode('.png', sheet)[1].tobytes())
    print('DONE saved=%d tried=%d out=%s' % (saved, tried, out), flush=True)


if __name__ == '__main__':
    main()
