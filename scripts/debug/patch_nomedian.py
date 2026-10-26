"""撤回中线墙：占据图恢复为“全部车道自由”。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

a = s.index('def scene_raster(world, frame):')
b = s.index('def build_scene(world):')
new = '''def scene_raster(world, frame):
    """模型的占据图：全部车道都算自由空间（不分方向、不加中线）。

    之前的"中线墙"实现会把路口画乱、还误伤路缘，已撤回。
    要限制逆行应该在车道级建图上做，不能用散点拼。
    """
    return lane_raster(world, frame, None)


'''
s = s[:a] + new + s[b:]
p.write_text(s, encoding='utf8')
print('reverted ok')
