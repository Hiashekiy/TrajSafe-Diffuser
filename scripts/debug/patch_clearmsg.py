"""净空检查输出“哪个障碍、在哪一段”，便于直接定位并删除。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

old = """    worst = 99.0
    for s in np.arange(0.0, path.s[-1], step):"""
new = """    worst = 99.0
    info = (None, None)
    for s in np.arange(0.0, path.s[-1], step):"""
assert old in s
s = s.replace(old, new, 1)

old = """        for P in polys:
            worst = min(worst, float(rect.distance(P)))
    return worst"""
new = """        for k, P in enumerate(polys):
            dist = float(rect.distance(P))
            if dist < worst:
                worst = dist
                info = (k, float(s))
    path_min_clearance.info = info
    return worst"""
assert old in s
s = s.replace(old, new, 1)

old = """    if clearance < 0.30:
        raise ValueError('绕不开：本车道的障碍把通道占满（净空 %.2f m）。\\n'
                         '可选：把终点挪到障碍前方、换个种子重抽、或允许借对向车道（关掉同向分组）'
                         % clearance)"""
new = """    if clearance < 0.30:
        k, ss = getattr(path_min_clearance, 'info', (None, None))
        who = '未知'
        if k is not None and k < len(STATE['obstacles']):
            o = STATE['obstacles'][k]
            who = '第 %d 个障碍（%s，位置 %.1f,%.1f）在轨迹 s=%.0f m 处' % (
                k+1, o.get('kind'), o['x'], o['y'], ss or 0.0)
        raise ValueError('绕不开：净空 %.2f m，最近的障碍是%s。\\n'
                         '可选：删掉该障碍、把终点挪到障碍前方、换个种子重抽、'
                         '或用「涂可行驶」把通道刷宽' % (clearance, who))"""
assert old in s
s = s.replace(old, new, 1)

p.write_text(s, encoding='utf8')
print('patched ok')
