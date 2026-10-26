"""加“候选路线可视化 + 手动选择”。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) FORCE 状态 + 候选缓存
s = s.replace("""SIDE = {'mode': 'auto', 'vec': None, 'orig': None, 'kept': None, 'total': None}""",
"""SIDE = {'mode': 'auto', 'vec': None, 'orig': None, 'kept': None, 'total': None}
FORCE = {'idx': None}          # 手动指定候选序号（None=交给模型选）""", 1)

s = s.replace("""    'side': 'auto',
}""", """    'side': 'auto',
    'cands': [],
}""", 1)

# 2) 候选过滤：支持强制序号
s = s.replace("""        keep = (lat > 0.4) if SIDE['mode'] == 'right' else (lat < -0.4)
        keep = keep & cands.mask""",
"""        keep = (lat > 0.4) if SIDE['mode'] == 'right' else (lat < -0.4)
        keep = keep & cands.mask
        if FORCE['idx'] is not None:
            vi = cands.valid_index()
            if len(vi):
                keep = np.zeros_like(cands.mask)
                keep[vi[int(FORCE['idx']) % len(vi)]] = True""", 1)

# 3) 候选生成函数（供接口和界面用）
s = s.replace("""def path_min_clearance(""",
"""def candidate_paths(world):
    \"\"\"只算骨架候选路线（不跑扩散），返回世界坐标折线列表。\"\"\"
    from src.carla_bridge.planner_adapter import PlannerAdapter
    from src.carla_bridge.skeleton_query import query_graph
    from src.geometry.skeleton_paths import generate_candidates
    d = install_candidate_bias() and None
    adapter = PlannerAdapter(cfg()['planner'])
    frame, occ, road, road_area, bodies = build_scene(world)
    cond = frame.scene(np.array([STATE['ego'][:2], STATE['goal'][:2]], float)).astype(np.float32)
    old = FORCE['idx']
    FORCE['idx'] = None
    graph = query_graph(adapter.engine.get_graph(adapter.engine.occupancy_key(occ), occ),
                        *cond)
    cands = generate_candidates(graph, cond[0], cond[1], adapter.online_cand_cfg)
    FORCE['idx'] = old
    out = []
    if cands.num_valid:
        for i in cands.valid_index():
            poly = cands.geometry[i] if i < len(cands.geometry) else cands.coords[i]
            out.append(np.asarray(frame.world(np.asarray(poly, float)), float))
    STATE['cands'] = out
    return out


def path_min_clearance(""", 1)

# 4) 左图画出候选（未生成轨迹时）
s = s.replace("""    plan = STATE['plan']
    if plan and plan['corridor']:""",
"""    if STATE.get('cands') and not STATE['plan']:
        cols = [(60, 200, 255), (255, 120, 60), (60, 255, 160), (200, 60, 255),
                (60, 120, 255), (255, 200, 60)]
        for i, poly in enumerate(STATE['cands']):
            g = np.rint(region_px(poly)*(1 << SHIFT)).astype(np.int32)
            cv2.polylines(img, [g], False, cols[i % len(cols)], 3, cv2.LINE_AA, shift=SHIFT)
            mid = poly[len(poly)//2]
            gp = region_px(np.array([mid]))[0]
            cv2.putText(img, str(i+1), (int(gp[0])+4, int(gp[1])-4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, cols[i % len(cols)], 2, cv2.LINE_AA)
    plan = STATE['plan']
    if plan and plan['corridor']:""", 1)

# 5) 接口：/api/candidates ；plan 接收 cand
s = s.replace("""                if self.path == '/api/state':""",
"""                if self.path == '/api/candidates':
                    if not (STATE['ego'] and STATE['goal']):
                        return self._json(dict(ok=False, msg='需要先设起点和终点'))
                    paths = candidate_paths(w)
                    return self._json(dict(ok=True, n=len(paths), occ=occ_png(w),
                                           view=view_png(w),
                                           msg='候选路线 %d 条（左图彩色线，数字为序号）' % len(paths)))
                if self.path == '/api/state':""", 1)

s = s.replace("""                    if p.get('side') in ('auto', 'left', 'right'):
                        STATE['side'] = p['side']""",
"""                    if p.get('side') in ('auto', 'left', 'right'):
                        STATE['side'] = p['side']
                    if 'cand' in p:
                        v = p['cand']
                        FORCE['idx'] = None if v in (None, '', -1, 'auto') else int(v)""", 1)

# 6) 界面：按钮 + 候选下拉
s = s.replace("""  <button onclick="saveScene()">保存场景</button>""",
"""  <button onclick="showCands()">看候选路线</button>
  <select id="cand" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px">
    <option value="auto">候选：自动</option>
  </select>
  <button onclick="saveScene()">保存场景</button>""", 1)

# 7) JS
s = s.replace("""async function saveScene(){""",
"""async function showCands(){
  if(!(st.ego&&st.goal)){ msg('先设起点和终点'); return; }
  msg('算候选路线中…');
  const s=await post('/api/candidates');
  if(s.occ) setImg('occ',s.occ);
  if(s.view) setImg('view',s.view);
  const sel=document.getElementById('cand');
  sel.innerHTML='<option value="auto">候选：自动</option>'+
    Array.from({length:s.n||0},(_,i)=>'<option value="'+(i+1)+'">候选 '+(i+1)+'</option>').join('');
  msg(s.msg||'');
}
async function saveScene(){""", 1)

s = s.replace("""  const body={side:document.getElementById('side').value};""",
"""  const body={side:document.getElementById('side').value,
              cand:document.getElementById('cand').value};""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
