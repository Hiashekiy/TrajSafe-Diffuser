"""走廊显示：先与路径±7 m 带求交，再【合并成一条轮廓】画，避免格子线糊成扇形。"""
from pathlib import Path

g = Path(__file__).resolve().parents[2]/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

start = s.index('            # 走廊：格子多边形会一直延伸到地图边界（扇状）')
end = s.index("            summary['corridor_cells_drawn'] = drawn") + len("            summary['corridor_cells_drawn'] = drawn")

new = '''            # 走廊：格子多边形会延伸到地图边界（扇状）→ 先与路径±7 m 带求交，再合并成一条轮廓
            from shapely.geometry import Polygon as _Poly, MultiPolygon as _Multi, LineString as _Line
            from shapely.ops import unary_union as _union
            band_clip = _Line(path.xy).buffer(7.0, cap_style=2, join_style=2)
            clipped_parts = []
            for poly in corridor:
                wp = _Poly(frame.world(np.asarray(poly)))
                if not wp.is_valid:
                    wp = wp.buffer(0)
                inter = wp.intersection(band_clip)
                if inter.is_empty:
                    continue
                clipped_parts.extend(inter.geoms if isinstance(inter, _Multi) else [inter])
            if clipped_parts:
                silhouette = _union(clipped_parts)
                parts = silhouette.geoms if isinstance(silhouette, _Multi) else [silhouette]
                layer = img.copy()
                for part in parts:
                    if part.is_empty:
                        continue
                    pts = project(np.asarray(part.exterior.coords)[:-1])
                    if pts is not None and len(pts) >= 3:
                        cv2.fillPoly(layer, [pts], (120, 215, 245))
                cv2.addWeighted(layer, 0.28, img, 0.72, 0, img)
                for part in parts:
                    if part.is_empty:
                        continue
                    pts = project(np.asarray(part.exterior.coords)[:-1])
                    if pts is not None and len(pts) >= 3:
                        cv2.polylines(img, [pts], True, (30, 140, 220), 2, cv2.LINE_AA)
                # 每 8 格画一条分隔线，示意"凸区域"仍是一格格拼的
                for k, part in enumerate(clipped_parts):
                    if k % 8:
                        continue
                    coords = np.asarray(part.exterior.coords)
                    ctr = coords.mean(axis=0)
                    pts = project(np.asarray(part.exterior.coords)[:-1])
                    if pts is not None and len(pts) >= 3:
                        cv2.polylines(img, [pts], True, (30, 140, 220), 1, cv2.LINE_AA)
            summary['corridor_cells_drawn'] = len(clipped_parts)'''

g.write_text(s[:start] + new + s[end:], encoding='utf8')
print('走廊改为合并轮廓')
