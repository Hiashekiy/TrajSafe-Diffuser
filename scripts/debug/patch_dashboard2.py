"""dashboard 大修：左右视图同朝向、按钮可用+高亮、换图即渲染、右侧套用论文风格轨迹。"""
from pathlib import Path

f = Path(__file__).resolve().parents[2] / 'scripts/dashboard_app.py'
t = f.read_text(encoding='utf8')

# ---------------------------------------------------------------- 1) 右侧渲染：论文风格 + 与左图同朝向/同中心
start = t.index('def render_join(')
end = t.index('# --------------------------------------------------------------------------- HTTP')
new_render = '''def render_join(world, frame, occ, road, path_xy=None):
    """右侧实景：与左图同朝向（场景坐标系），论文风格车身带 + 中心线 + 自车/障碍框。"""
    import cv2
    import carla
    from src.carla_bridge.path_profile import Path as GeomPath

    ego = STATE['ego']
    half_w = OBSTACLE_DIMS['car'][1]
    yaw_deg = float(np.rad2deg(frame.yaw))          # 与左图完全一致的朝向

    if path_xy is not None and len(path_xy) > 5:
        geom = GeomPath.build(path_xy)
        centre = np.array(geom.pose(geom.s[-1]*0.5)[0], float)
        span = float(geom.s[-1])+22.0
    else:
        geom = None
        centre = np.array(ego[:2], float)
        span = 70.0
    height = span/1.125

    bp = world.get_blueprint_library().find('sensor.camera.rgb')
    bp.set_attribute('image_size_x', '1100')
    bp.set_attribute('image_size_y', '1100')
    bp.set_attribute('fov', '90')
    cam = world.spawn_actor(bp, carla.Transform(
        carla.Location(x=float(centre[0]), y=float(centre[1]), z=60+height),
        carla.Rotation(pitch=-90.0, yaw=yaw_deg)))
    frames = []
    cam.listen(lambda img: frames.append(
        (np.frombuffer(img.raw_data, np.uint8).reshape(img.height, img.width, 4)[:, :, :3].copy(),
         np.asarray(img.transform.get_inverse_matrix(), float))))
    settings = world.get_settings()
    was_sync = settings.synchronous_mode
    settings.synchronous_mode = True
    settings.fixed_delta_seconds = 0.05
    world.apply_settings(settings)
    for _ in range(4):
        world.tick()
    rgb, inv = frames[-1] if frames else (np.zeros((1100, 1100, 3), np.uint8), np.eye(4))
    settings.synchronous_mode = was_sync
    world.apply_settings(settings)
    try:
        cam.destroy()
    except Exception:
        pass

    h, w = rgb.shape[:2]
    ground = 0.06

    def px(points):
        pts = np.asarray(points, float).reshape(-1, 2)
        hom = np.column_stack([pts, np.full(len(pts), ground), np.ones(len(pts))])
        c3 = hom @ inv.T
        c3[:, 0] = np.maximum(c3[:, 0], 0.05)
        fpx = w/2.0
        return np.column_stack([w/2+fpx*c3[:, 1]/c3[:, 0], h/2-fpx*c3[:, 2]/c3[:, 0]])

    SHIFT, SCALE = 4, 16

    def pxs(points):
        return np.rint(px(points)*SCALE).astype(np.int32)

    # 记录单应矩阵（右图点击用）
    try:
        probe_w = np.array([[centre[0], centre[1]], [centre[0]+15, centre[1]],
                            [centre[0], centre[1]+15], [centre[0]+15, centre[1]+15]], np.float32)
        STATE['view_H'] = cv2.getPerspectiveTransform(probe_w, px(probe_w).astype(np.float32)).tolist()
        STATE['view_size'] = [int(w), int(h)]
    except Exception:
        STATE['view_H'] = None

    img = rgb.astype(np.float32)
    if geom is not None:
        half = half_w+0.28
        left, right = [], []
        for s in np.arange(0.0, geom.s[-1], 0.4):
            p, yaw = geom.pose(float(s))
            n = np.array([-np.sin(yaw), np.cos(yaw)])
            left.append(p+n*half)
            right.append(p-n*half)
        band = np.vstack([np.array(left), np.array(right)[::-1]])
        for k in range(12):                                     # 柔和边缘
            t_k = k/11.0
            bb = np.vstack([np.array(left)*(1+0.0)+np.array(left)*(0.0),
                            np.array(right)]) if False else band
            scale = 1.0+0.40*(1.0-t_k)
            mid = np.array(left).mean(0)
            bb = (band-mid)*scale+mid
            mask = np.zeros((h, w), np.float32)
            cv2.fillPoly(mask, [pxs(bb)], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
            mask = cv2.GaussianBlur(mask, (0, 0), 3.0)[:, :, None]
            img = img*(1-0.035*mask)+np.array((80, 220, 90), np.float32)*0.035*mask
        mask = np.zeros((h, w), np.float32)
        cv2.fillPoly(mask, [pxs(band)], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
        mask = cv2.GaussianBlur(mask, (0, 0), 3.0)[:, :, None]
        img = img*(1-0.30*mask)+np.array((80, 220, 90), np.float32)*0.30*mask
        core = np.array(left), np.array(right)
        core_band = np.vstack([core[0]*0.62+centre*0.38, (core[1]*0.62+centre*0.38)[::-1]])
        maskc = np.zeros((h, w), np.float32)
        cv2.fillPoly(maskc, [pxs(core_band)], 1.0, shift=SHIFT, lineType=cv2.LINE_AA)
        maskc = cv2.GaussianBlur(maskc, (0, 0), 3.0)[:, :, None]
        img = img*(1-0.22*maskc)+np.array((150, 255, 170), np.float32)*0.22*maskc
        line = pxs(path_xy)
        cv2.polylines(img, [line], False, (240, 252, 246), 2, cv2.LINE_AA, shift=SHIFT)

    out = np.clip(img, 0, 255).astype(np.uint8)
    hl, hw = OBSTACLE_DIMS['car']
    c = np.array(ego[:2], float)
    yaw = np.deg2rad(ego[2])
    d = np.array([np.cos(yaw), np.sin(yaw)])
    n = np.array([-np.sin(yaw), np.cos(yaw)])
    box = np.array([c+d*hl+n*hw, c+d*hl-n*hw, c-d*hl-n*hw, c-d*hl+n*hw])
    cv2.polylines(out, [pxs(box)], True, (255, 200, 60), 2, cv2.LINE_AA, shift=SHIFT)
    for o in STATE['obstacles']:
        ohl, ohw = OBSTACLE_DIMS.get(o['kind'], OBSTACLE_DIMS['car'])
        oc = np.array([o['x'], o['y']])
        oy = np.deg2rad(o['yaw'])
        od = np.array([np.cos(oy), np.sin(oy)])
        on = np.array([-np.sin(oy), np.cos(oy)])
        ob = np.array([oc+od*ohl+on*ohw, oc+od*ohl-on*ohw,
                       oc-od*ohl-on*ohw, oc-od*ohl+on*ohw])
        cv2.polylines(out, [pxs(ob)], True, (60, 60, 235), 2, cv2.LINE_AA, shift=SHIFT)
    ok, buf = cv2.imencode('.png', out)
    return base64.b64encode(buf.tobytes()).decode()


'''
t = t[:start] + new_render + t[end:]

# ---------------------------------------------------------------- 2) 前端：按钮高亮 + 换图自动出右图
html_start = t.index('HTML = """')
html_end = t.index('"""\n\n\ndef pick_random_ego')
new_html = '''HTML = """<!doctype html><html><head><meta charset="utf-8"><title>TrajSafe 场景工具</title>
<style>
body{font-family:system-ui,Segoe UI,sans-serif;margin:0;background:#0f1013;color:#eee}
#bar{padding:8px 12px;background:#1b1c21;display:flex;gap:8px;align-items:center;flex-wrap:wrap;
     position:sticky;top:0;z-index:9;border-bottom:1px solid #2a2c33}
button{background:#33363f;border:1px solid #444a55;color:#e8e8e8;padding:7px 13px;border-radius:6px;
       cursor:pointer;font-size:13px}
button:hover{background:#3d4插1}
button.active{background:#2d7ef7;border-color:#2d7ef7;color:#fff}
button.primary{background:#1f9d55;border-color:#1f9d55;color:#fff}
button:disabled{opacity:.45;cursor:default}
#msg{color:#8ecdf7;font-size:12.5px;margin-left:6px}
#wrap{display:flex;gap:10px;padding:10px;align-items:flex-start}
.pane{background:#15171c;border:1px solid #23262d;border-radius:8px;padding:6px;flex:1}
.pane h4{margin:2px 0 6px;font-size:12.5px;color:#8b93a1;font-weight:600}
canvas{width:100%;display:block;border-radius:4px;background:#000;cursor:crosshair}
</style></head><body>
<div id="bar">
  <b style="margin-right:6px">TrajSafe 场景工具</b>
  <button onclick="randomMap(this)">随机地图/起点</button>
  <button class="tool active" data-mode="ego" onclick="setMode(this)">设置起点</button>
  <button class="tool" data-mode="car" onclick="setMode(this)">放车辆</button>
  <button class="tool" data-mode="cone" onclick="setMode(this)">放锥桶</button>
  <button class="tool" data-mode="barrier" onclick="setMode(this)">放施工栏</button>
  <button onclick="undoObstacle()">撤销障碍</button>
  <button class="primary" id="gen" onclick="plan()">生成轨迹</button>
  <span id="msg">点左图设置起点 / 放障碍</span>
</div>
<div id="wrap">
  <div class="pane"><h4>占据栅格（全部车道，实时）</h4><canvas id="occ"></canvas></div>
  <div class="pane"><h4>CARLA 实景 + 轨迹（同朝向）</h4><canvas id="view"></canvas></div>
</div>
<script>
let addMode='ego', occImg=null, viewImg=null;
const cOcc=document.getElementById('occ'), cView=document.getElementById('view');
const gOcc=cOcc.getContext('2d'), gView=cView.getContext('2d');
function draw(){
  if(occImg){cOcc.width=occImg.width;cOcc.height=occImg.height;gOcc.drawImage(occImg,0,0);}
  if(viewImg){cView.width=viewImg.width;cView.height=viewImg.height;gView.drawImage(viewImg,0,0);}
}
function setImg(id,b64){const im=new Image();im.onload=()=>{if(id==='occ')occImg=im;else viewImg=im;draw();};im.src='data:image/png;base64,'+b64;}
async function post(u,b){
  const r=await fetch(u,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(b||{})});
  return r.json();
}
function msg(s){document.getElementById('msg').textContent=s;}
function setMode(btn){
  addMode=btn.dataset.mode;
  document.querySelectorAll('.tool').forEach(b=>b.classList.remove('active'));
  btn.classList.add('active');
  msg('当前模式：'+btn.textContent+'　点左图或右图放置');
}
async function refresh(){
  const s=await post('/api/occupancy');
  if(s.occ) setImg('occ',s.occ);
  msg('障碍 '+s.n_obs+' 个 ｜ 起点 '+s.ego.join(', '));
}
async function randomMap(btn){
  if(btn) btn.disabled=true;
  msg('切换地图中…');
  await post('/api/map');
  viewImg=null;
  await refresh();
  await plan();                     // 换图后直接渲染右侧
  if(btn) btn.disabled=false;
}
async function undoObstacle(){ await post('/api/undo'); await refresh(); }
async function plan(){
  const g=document.getElementById('gen'); g.disabled=true; msg('规划中…');
  const s=await post('/api/plan');
  if(s.occ) setImg('occ',s.occ);
  if(s.view) setImg('view',s.view);
  msg(s.msg||'');
  g.disabled=false;
}
function clickPos(canvas,ev){
  const r=canvas.getBoundingClientRect();
  return {u:(ev.clientX-r.left)/r.width, v:(ev.clientY-r.top)/r.height};
}
cOcc.onclick=async e=>{const p=clickPos(cOcc,e);await post('/api/click',{mode:addMode,u:p.u,v:p.v});await refresh();};
cView.onclick=async e=>{const p=clickPos(cView,e);await post('/api/click',{mode:addMode,u:p.u,v:p.v,side:'view'});await refresh();};
refresh();
</script></body></html>'''
t = t[:html_start] + new_html + t[html_end:]

f.write_text(t, encoding='utf8')
print('dashboard: 前端按钮/自动渲染 + 右图论文风格 + 同朝向 已重写')
