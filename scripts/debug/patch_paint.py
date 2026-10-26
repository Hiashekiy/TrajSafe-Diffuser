"""加“涂抹/擦除”笔刷：用户的涂抹直接叠加到模型输入栅格上。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) 状态
s = s.replace("""    'cands': [],
}""", """    'cands': [],
    'paint': [],          # [{mode:'block'|'free', r:米, pts:[[x,y],...]}]
}""", 1)

# 2) 涂抹应用函数 + 接进 scene_raster
s = s.replace("""def scene_raster(world, frame):""",
"""def apply_paint(mask, frame):
    \"\"\"把用户涂抹按顺序盖到栅格上（block=1 不可行驶，free=0 可行驶）。\"\"\"
    import cv2
    cell = frame.size/frame.resolution
    for stroke in STATE['paint']:
        rr = max(1, int(round(float(stroke['r'])/cell)))
        val = 1 if stroke['mode'] == 'block' else 0
        g = frame.grid(np.asarray(stroke['pts'], float))
        for cx, cy in g:
            cv2.circle(mask, (int(round(cx)), int(round(cy))), rr, val, -1)
    return mask


def scene_raster(world, frame):""", 1)

s = s.replace("""    return lane_raster(world, frame, None)


def build_scene""",
"""    return apply_paint(lane_raster(world, frame, None), frame)


def build_scene""", 1)

# 3) 接口
s = s.replace("""                if self.path == '/api/candidates':""",
"""                if self.path == '/api/paint':
                    mode = 'block' if p.get('mode') == 'block' else 'free'
                    r = float(p.get('r', 2.0))
                    pts = []
                    for uv in (p.get('pts') or []):
                        x, y = region_world(float(uv[0]), float(uv[1]))
                        pts.append([x, y])
                    if pts:
                        STATE['paint'].append(dict(mode=mode, r=r, pts=pts))
                        STATE['plan'] = None
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='涂抹 %d 笔（%s）' % (
                                               len(STATE['paint']),
                                               '不可行驶' if mode == 'block' else '可行驶')))
                if self.path == '/api/paint_undo':
                    if STATE['paint']:
                        STATE['paint'].pop()
                        STATE['plan'] = None
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'],
                                           msg='撤销一笔，剩 %d 笔' % len(STATE['paint'])))
                if self.path == '/api/paint_clear':
                    STATE['paint'] = []
                    STATE['plan'] = None
                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg='已清空涂抹'))
                if self.path == '/api/candidates':""", 1)

# 4) 工具栏
s = s.replace("""  <button id="cropbtn" onclick="toggleCrop()">截图</button>""",
"""  <button id="paintBlock" onclick="setPaint('block',this)">涂不可行驶</button>
  <button id="paintFree" onclick="setPaint('free',this)">涂可行驶</button>
  <select id="brush" style="background:#33363f;color:#e8e8e8;border:1px solid #444a55;
          border-radius:6px;padding:7px 8px;font-size:13px">
    <option value="1.5">笔刷 1.5 m</option>
    <option value="2.5" selected>笔刷 2.5 m</option>
    <option value="4">笔刷 4 m</option>
    <option value="6">笔刷 6 m</option>
  </select>
  <button onclick="paintUndo()">撤销涂抹</button>
  <button onclick="paintClear()">清空涂抹</button>
  <button id="cropbtn" onclick="toggleCrop()">截图</button>""", 1)

# 5) JS
s = s.replace("""let cropMode=false,""",
"""let paintMode=null, painting=false, strokePts=[];
function setPaint(m,btn){
  paintMode=(paintMode===m)?null:m;
  cropMode=false;
  document.getElementById('paintBlock').style.background = paintMode==='block'?'#c02d2d':'#33363f';
  document.getElementById('paintFree').style.background  = paintMode==='free'?'#1f7a3f':'#33363f';
  msg(paintMode? '涂抹模式：在左图上拖动（'+document.getElementById('brush').value+' m 笔刷）' : '已退出涂抹');
}
function paintUV(c,ev){
  const p=canvasPos(c,ev);
  return [p.x/c.width, p.y/c.height, p];
}
function paintDown(c,ev){
  if(!paintMode) return false;
  painting=true; strokePts=[paintUV(c,ev)]; c.__paintPane=c; return true;
}
function paintMove(c,ev){
  if(!painting||!paintMode) return;
  const t=paintUV(c,ev); strokePts.push(t);
  const g=(c===cOcc)?gOcc:gView;
  const rp=(+document.getElementById('brush').value)/(160/256)*(c.width/256);
  g.save(); g.fillStyle=(paintMode==='block')?'rgba(230,60,60,0.35)':'rgba(120,255,150,0.30)';
  g.beginPath(); g.arc(t[2].x,t[2].y,rp,0,6.2832); g.fill(); g.restore();
}
async function paintUp(){
  if(!painting||!paintMode){ painting=false; return; }
  painting=false;
  const pts=strokePts.map(t=>[t[0],t[1]]);
  strokePts=[];
  if(pts.length<2) return;
  apply(await post('/api/paint',{mode:paintMode, r:+document.getElementById('brush').value, pts:pts}));
}
async function paintUndo(){ apply(await post('/api/paint_undo')); }
async function paintClear(){ if(!confirm('清空所有涂抹？')) return; apply(await post('/api/paint_clear')); }

let cropMode=false,""", 1)

# 6) 鼠标事件
s = s.replace("""cOcc.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cOcc,e); return;} });""",
"""cOcc.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cOcc,e); return;}
  if(paintDown(cOcc,e)){e.preventDefault();} });
cOcc.addEventListener('mousemove',e=>paintMove(cOcc,e));""", 1)
s = s.replace("""cView.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cView,e); return;} });""",
"""cView.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cView,e); return;}
  if(paintDown(cView,e)){e.preventDefault();} });
cView.addEventListener('mousemove',e=>paintMove(cView,e));""", 1)
s = s.replace("""window.addEventListener('mouseup',cordUp);""",
"""window.addEventListener('mouseup',cordUp);
window.addEventListener('mouseup',paintUp);""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
