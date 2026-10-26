"""① runner 只存干净俯视原图 + 几何数据；② 另写脚本出图。"""
from pathlib import Path

root = Path(__file__).resolve().parents[2]
g = root/'scripts/carla_scenario_sim.py'
s = g.read_text(encoding='utf8')

old = '''        np.savez_compressed(
            out/'plan.npz', curve_world=path.xy, controls=result['controls'],'''
new = '''        # 俯视原图 + 相机矩阵留给 render_figure.py 出图（CARLA 只负责"拍照"）
        if captured:
            rgb_last, inv_last = captured[-1]
            np.savez_compressed(out/'view.npz', rgb=rgb_last, inverse=inv_last,
                                ground_z=float(z), half_width=float(ego_body.half_width),
                                obstacle_corners=np.array([o.corners(0.0) for o in obstacle_bodies],
                                                          dtype=object))
        np.savez_compressed(
            out/'plan.npz', curve_world=path.xy, controls=result['controls'],'''
assert old in s
s = s.replace(old, new)
g.write_text(s, encoding='utf8')
print('runner: 额外保存 view.npz（原图+矩阵+几何）')
