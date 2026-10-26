"""在失败点复现 build_scene，导出模型栅格看问题。"""
import sys
import numpy as np
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT/'scripts'))
sys.path.insert(0, 'E:/CARLA_0.9.16/PythonAPI/carla')
import cv2
import dashboard_app as d

EGO = [-52.1, 82.4, 89.8, 0.0]
GOAL = [-51.7, 140.4, 0.0]
d.STATE['ego'] = EGO
d.STATE['goal'] = GOAL
d.STATE['region'] = dict(cx=-52.1, cy=82.7)

w = d.carla_world()
frame, occ, road, road_area, bodies, chain, i_ego = d.build_scene(w, np.array(EGO[:2]))
print('chain', len(chain), 'i_ego', i_ego, 'free', float((occ == 0).mean()).__round__(3))
print('goal scene', frame.scene(np.array(GOAL[:2]))[0].round(3))
print('goal free?', frame.free(occ, np.array([GOAL[:2]]))[0])
print('ego free?', frame.free(occ, np.array([EGO[:2]]))[0])

img = np.zeros((frame.resolution, frame.resolution, 3), np.uint8)
img[road == 1] = (70, 70, 70)
img[occ == 0] = (235, 235, 235)
g = frame.grid(np.array(GOAL[:2])).astype(int)
cv2.drawMarker(img, tuple(g), (60, 60, 235), cv2.MARKER_TILTED_CROSS, 11, 2)
cv2.circle(img, tuple(frame.grid(np.array(EGO[:2])).astype(int)), 4, (40, 140, 240), -1)
out = cv2.resize(img, (768, 768), interpolation=cv2.INTER_NEAREST)
cv2.imwrite(str(ROOT/'outputs'/'probe_model.png'), out)
print('wrote outputs/probe_model.png')
