"""显示裁剪：走廊格子多边形与“路径±7 m 缓冲带”求交，画出干净的管状走廊。"""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

old = '''            for i, poly in enumerate(corridor):                 # 走廊：逐格交替色 + 描边
                pts = project(frame.world(np.asarray(poly)))
                if pts is None:
                    continue
                fill = (150, 210, 255) if i % 2 else (120, 235, 215)
                layer = img.copy()
                cv2.fillPoly(layer, [pts], fill)
                cv2.addWeighted(layer, 0.22, img, 0.78, 0, img)
                cv2.polylines(img, [pts], True, (60, 150, 220), 1, cv2.LINE_AA)'''

new = '''            # 走廊：格子多边形会一直延伸到地图边界（扇状），显示时裁到路径±7 m 带内
            from shapely.geometry import Polygon as _Poly, MultiPolygon as _Multi, LineString as _Line
            band_clip = _Line(path.xy).buffer(7.0, cap_style=2, join_style=2)
            drawn = 0
            for poly in corridor:
                world_poly = _Poly(frame.world(np.asarray(poly)))
                if not world_poly.is_valid:
                    world_poly = world_poly.buffer(0)
                clipped = world_poly.intersection(band_clip)
                parts = clipped.geoms if isinstance(clipped, _Multi) else [clipped]
                for part in parts:
                    if part.is_empty or part.area < 0.2:
                        continue
                    pts = project(np.asarray(part.exterior.coords)[:-1])
                    if pts is None or len(pts) < 3:
                        continue
                    layer = img.copy()
                    cv2.fillPoly(layer, [pts], (120, 215, 245))
                    cv2.addWeighted(layer, 0.26, img, 0.74, 0, img)
                    cv2.polylines(img, [pts], True, (40, 150, 225), 1, cv2.LINE_AA)
                    drawn += 1
            summary['corridor_cells_drawn'] = drawn'''

assert old in s
g.write_text(s.replace(old, new), encoding='utf8')
print('走廊显示裁剪已加')
