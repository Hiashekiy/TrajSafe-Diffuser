"""给 dashboard 加“绕行侧 自动/左/右”开关（只过滤候选，不动模型和走廊）。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) 插入候选过滤器（放在 relax_anchors 之前）
anchor = "ANCHOR = {'relaxed': False}"
code = '''SIDE = {'mode': 'auto', 'vec': None, 'orig': None, 'kept': None, 'total': None}


def install_candidate_bias():
    """绕行侧过滤：候选生成后，按“候选相对车头的平均横向偏移符号”筛掉另一侧。

    只影响交��模型挑哪条候选，不改模型、不改走廊、不改栅格。
    指定的一侧一条候选都没有时，退回不过滤（不会把功能搞死）。
    """
    import src.geometry.skeleton_paths as sp
    if SIDE['orig'] is not None:
        return
    SIDE['orig'] = sp.generate_candidates

    def wrapped(graph, start, goal, cfg):
        cands = SIDE['orig'](graph, start, goal, cfg)
        vec = SIDE['vec']
        if vec is None or cands.num_valid == 0:
            return cands
        start = np.asarray(start, float)
        vec = np.asarray(vec, float)
        lat = np.zeros(cands.num_slots)
        for i in range(cands.num_slots):
            if not cands.mask[i]:
                continue
            poly = cands.geometry[i] if i < len(cands.geometry) else cands.coords[i]
            poly = np.asarray(poly, float)
            lat[i] = float(np.mean((poly-start) @ vec))
        keep = (lat > 0.4) if SIDE['mode'] == 'right' else (lat < -0.4)
        keep = keep & cands.mask
        SIDE['total'] = int(cands.num_valid)
        if not keep.any():
            SIDE['kept'] = None
            return cands
        cands.mask = keep
        SIDE['kept'] = int(keep.sum())
        return cands

    sp.generate_candidates = wrapped


'''
s = s.replace(anchor, code+anchor, 1)

# 2) plan() 里设置过滤参数
old = """    adapter = PlannerAdapter(c['planner'])
    t0 = time.perf_counter()"""
new = """    install_candidate_bias()
    SIDE['mode'] = STATE.get('side', 'auto')
    if SIDE['mode'] == 'auto':
        SIDE['vec'] = None
    else:
        h = np.deg2rad(ego_yaw)
        SIDE['vec'] = np.array([-np.sin(h), np.cos(h)])   # 车头右侧单位向量
    SIDE['kept'] = SIDE['total'] = None

    adapter = PlannerAdapter(c['planner'])
    t0 = time.perf_counter()"""
s = s.replace(old, new, 1)

# 3) 提示栏带上绕行侧信息
old = """    if check is not None:
        note += ' ｜ 越界 %s m ｜ 最小间隙 %s m ｜ %s' % ("""
new = """    if SIDE['mode'] != 'auto':
        note += ' ｜ 绕行侧 %s（候选 %s/%s）' % (
            '右' if SIDE['mode'] == 'right' else '左',
            SIDE['kept'] if SIDE['kept'] is not None else '同侧为空',
            SIDE['total'])
    if check is not None:
        note += ' ｜ 越界 %s m ｜ 最小间隙 %s m ｜ %s' % ("""
s = s.replace(old, new, 1)

# 4) STATE 增加 side 字段
s = s.replace("""    'seed': 43,
}""", """    'seed': 43,
    'side': 'auto',
}""", 1)

# 5) 服务端接收 side
old = """                    seed = p.get('seed')"""
new = """                    if p.get('side') in ('auto', 'left', 'right'):
                        STATE['side'] = p['side']
                    seed = p.get('seed')"""
s = s.replace(old, new, 1)

# 6) 界面：绕行侧下拉框
old = """  <button onclick="saveScene()">保存场景</button>"""
new = """  <select id="side" onchange="STATE_side=this.value" title="绕行侧"
          style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px">
    <option value="auto">绕行：自动</option>
    <option value="left">绕行：左</option>
    <option value="right">绕行：右</option>
  </select>
  <button onclick="saveScene()">保存场景</button>"""
s = s.replace(old, new, 1)

# 7) JS：带上 side
s = s.replace("""  const body={}; if(seed!==undefined) body.seed=seed;""",
              """  const body={side:document.getElementById('side').value};
  if(seed!==undefined) body.seed=seed;""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
