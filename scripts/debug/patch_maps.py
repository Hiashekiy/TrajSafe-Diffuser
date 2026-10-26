"""加地图选择：/api/maps 列出，/api/load_map 切换。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) pick_region：只有 Town10 用已验证点位，其它地图随机挑并把比例做参考
s = s.replace("""def pick_region(world):
    \"\"\"随机地块：从已验证的直路段里轮换（路口/急弯处候选通道生成不出来）。\"\"\"
    sps = world.get_map().get_spawn_points()""",
"""def pick_region(world):
    \"\"\"随机地块：Town10 用已验证的直路段；其它地图随机挑（按路面占比择优）。\"\"\"
    sps = world.get_map().get_spawn_points()
    if 'Town10' not in world.get_map().name:
        rng = np.random.default_rng()
        best = None
        for _ in range(10):
            sp = sps[int(rng.integers(len(sps)))]
            STATE['region'] = dict(cx=float(sp.location.x), cy=float(sp.location.y))
            ratio = float(region_road(world).mean())
            if best is None or ratio > best[0]:
                best = (ratio, dict(STATE['region']))
            if ratio >= 0.08:
                break
        STATE['region'] = best[1]
        return -1""", 1)

# 2) 切图缓存失效 + 换图函数
s = s.replace("""def snapshot():""",
"""def invalidate_caches():
    TOPO['map'], TOPO['segments'] = None, None
    CAM['key'], CAM['rgb'], CAM['inv'] = None, None, None
    MEDIAN['map'], MEDIAN['sign'], MEDIAN['lines'] = None, None, None
    CYCLE['order'], CYCLE['i'] = None, 0


def snapshot():""", 1)

# 3) 接口
s = s.replace("""                if self.path == '/api/map':""",
"""                if self.path == '/api/maps':
                    import carla
                    cl = carla.Client('127.0.0.1', 2000)
                    cl.set_timeout(20.0)
                    try:
                        names = sorted(cl.get_available_maps())
                    except Exception:
                        names = [STATE['map']]
                    return self._json(dict(ok=True, maps=[n.rsplit('/', 1)[-1] for n in names],
                                           cur=STATE['map'].rsplit('/', 1)[-1]))
                if self.path == '/api/load_map':
                    import carla
                    name = str(p.get('name') or '').strip()
                    if not name:
                        return self._json(dict(ok=False, msg='未指定地图'))
                    cl = carla.Client('127.0.0.1', 2000)
                    cl.set_timeout(60.0)
                    try:
                        cl.load_world(name)
                    except Exception as exc:
                        return self._json(dict(ok=False, msg='加载地图失败：%s' % exc))
                    w2 = cl.get_world()
                    holder['world'] = w2
                    STATE.update(ego=None, goal=None, obstacles=[], plan=None, paint=[], cands=[])
                    STATE['map'] = w2.get_map().name
                    invalidate_caches()
                    pick_region(w2)
                    respawn(w2)
                    return self._json(dict(ok=True, region=STATE['region'],
                                           occ=occ_png(w2), view=view_png(w2),
                                           ego=None, goal=None, map=STATE['map'],
                                           msg='已加载 %s ｜ 点起点开始' % STATE['map'].rsplit('/', 1)[-1]))
                if self.path == '/api/map':""", 1)

# 4) /api/map 也失效缓存（换地块后相机/拓扑按新地图名仍有效，但相机必须重拍）
s = s.replace("""                if self.path == '/api/map':
                    STATE.update(ego=None, goal=None, obstacles=[], plan=None)""",
"""                if self.path == '/api/map':
                    STATE.update(ego=None, goal=None, obstacles=[], plan=None, cands=[])""", 1)

# 5) 界面：地图下拉 + 按钮
s = s.replace("""  <button onclick="newMap(this)">切换地图</button>""",
"""  <select id="maps" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px;max-width:150px"></select>
  <button onclick="loadMap(this)">加载地图</button>
  <button onclick="newMap(this)">换个地块</button>""", 1)

s = s.replace("""async function newMap(btn){""",
"""async function refreshMaps(){
  const s=await post('/api/maps');
  const sel=document.getElementById('maps');
  if(!sel||!s.maps) return;
  sel.innerHTML=s.maps.map(m=>'<option'+(m===s.cur?' selected':'')+'>'+m+'</option>').join('');
}
async function loadMap(btn){
  const n=document.getElementById('maps').value;
  if(!n) return;
  btn.disabled=true; msg('正在加载 '+n+'（10~40 秒）…');
  apply(await post('/api/load_map',{name:n}));
  btn.disabled=false;
}
async function newMap(btn){""", 1)

s = s.replace("""refreshScenes();""", """refreshScenes();\nrefreshMaps();""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
