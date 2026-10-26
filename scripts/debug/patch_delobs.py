"""加“点选删除某个障碍 / 清空障碍”。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) 工具栏：删障碍模式 + 清空
s = s.replace("""  <button onclick="undo()">撤销障碍</button>""",
"""  <button class="tool" data-mode="del" onclick="setMode(this)">删障碍</button>
  <button onclick="undo()">撤销障碍</button>
  <button onclick="clearObs()">清空障碍</button>""", 1)

# 2) /api/click 支持 del
s = s.replace("""                    if mode == 'ego':
                        STATE['ego'] = [float(x), float(y), yaw, z]""",
"""                    if mode == 'del':
                        if not STATE['obstacles']:
                            return self._json(dict(ok=False, msg='当前没有障碍'))
                        d2 = [np.hypot(o['x']-x, o['y']-y) for o in STATE['obstacles']]
                        k = int(np.argmin(d2))
                        if d2[k] > 4.0:
                            return self._json(dict(ok=False,
                                                   msg='离最近的障碍 %.1f m，点近一点' % d2[k]))
                        STATE['obstacles'].pop(k)
                        STATE['plan'] = None
                        respawn(w)
                        return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                               ego=STATE['ego'], goal=STATE['goal'],
                                               region=STATE['region'],
                                               msg='已删除第 %d 个障碍 ｜ %s' % (k+1, status())))
                    if mode == 'ego':
                        STATE['ego'] = [float(x), float(y), yaw, z]""", 1)

# 3) /api/clear_obstacles
s = s.replace("""                if self.path == '/api/undo':""",
"""                if self.path == '/api/clear_obstacles':
                    STATE['obstacles'] = []
                    STATE['plan'] = None
                    respawn(w)
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='已清空障碍 ｜ '+status()))
                if self.path == '/api/undo':""", 1)

# 4) JS
s = s.replace("""async function undo(){ apply(await post('/api/undo')); }""",
"""async function undo(){ apply(await post('/api/undo')); }
async function clearObs(){
  if(!confirm('清空所有障碍？')) return;
  apply(await post('/api/clear_obstacles'));
}""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
