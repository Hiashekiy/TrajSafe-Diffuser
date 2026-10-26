"""加“截图”：拖框 → 按正方形裁切 → 下载 PNG。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

# 1) 按钮
s = s.replace("""  <button onclick="showCands()">看候选路线</button>""",
"""  <button id="cropbtn" onclick="toggleCrop()">截图</button>
  <button onclick="showCands()">看候选路线</button>""", 1)

# 2) JS：裁剪逻辑（插在 clicked 之前）
s = s.replace("""async function clicked(canvas,ev){""",
"""let cropMode=false, cropPane=null, cropStart=null, cropRect=null;
function toggleCrop(){
  cropMode=!cropMode; cropRect=null; cropStart=null; cropPane=null;
  const b=document.getElementById('cropbtn');
  b.style.background=cropMode?'#c02d2d':'#33363f';
  b.style.borderColor=cropMode?'#c02d2d':'#444a55';
  msg(cropMode?'截图模式：在任意一张图上拖出选框（自动裁成正方形）':'已退出截图模式');
  draw();
}
function canvasPos(c,ev){
  const r=c.getBoundingClientRect();
  return {x:(ev.clientX-r.left)*c.width/r.width, y:(ev.clientY-r.top)*c.height/r.height};
}
function squareRect(a,b,W,H){
  const side=Math.max(Math.abs(b.x-a.x),Math.abs(b.y-a.y),48);
  let cx=(a.x+b.x)/2, cy=(a.y+b.y)/2;
  let x=cx-side/2, y=cy-side/2;
  x=Math.min(Math.max(x,0),Math.max(0,W-side));
  y=Math.min(Math.max(y,0),Math.max(0,H-side));
  return {x:x, y:y, side:Math.min(side,Math.min(W,H))};
}
function saveCrop(c,rect,pane){
  const out=document.createElement('canvas');
  out.width=Math.round(rect.side); out.height=Math.round(rect.side);
  out.getContext('2d').drawImage(c, rect.x, rect.y, rect.side, rect.side,
                                 0, 0, out.width, out.height);
  const a=document.createElement('a');
  const tag=(pane==='occ')?'occupancy':'carla';
  a.download='crop_'+tag+'_'+Date.now()+'.png';
  a.href=out.toDataURL('image/png');
  a.click();
  msg('已下载正方形截图 '+out.width+'×'+out.height+'（'+tag+'）');
}
function cordSel(c,ev){
  if(!cropMode) return;
  cropPane=c; cropStart=canvasPos(c,ev); cropRect=null; draw();
}
function cordMove(ev){
  if(!cropMode||!cropStart||!cropPane) return;
  const p=canvasPos(cropPane,ev);
  cropRect=squareRect(cropStart,p,cropPane.width,cropPane.height);
  draw();
}
function cordUp(){
  if(!cropMode||!cropStart||!cropPane) return;
  const c=cropPane, r=cropRect;
  cropStart=null; cropPane=null; cropRect=null;
  if(r&&r.side>=48) saveCrop(c,r,c===cOcc?'occ':'view');
  draw();
}
window.addEventListener('mousemove',cordMove);
window.addEventListener('mouseup',cordUp);

async function clicked(canvas,ev){
  if(cropMode) return;
""", 1)

# 3) draw() 里画选框
s = s.replace("""function draw(){
  if(occImg){cOcc.width=occImg.width;cOcc.height=occImg.height;gOcc.drawImage(occImg,0,0);}
  if(viewImg){cView.width=viewImg.width;cView.height=viewImg.height;gView.drawImage(viewImg,0,0);}
}""",
"""function draw(){
  if(occImg){cOcc.width=occImg.width;cOcc.height=occImg.height;gOcc.drawImage(occImg,0,0);}
  if(viewImg){cView.width=viewImg.width;cView.height=viewImg.height;gView.drawImage(viewImg,0,0);}
  if(cropRect){
    const g=(cropPane===cOcc)?gOcc:gView;
    g.save();
    g.strokeStyle='#ff2d55'; g.lineWidth=2;
    g.strokeRect(cropRect.x, cropRect.y, cropRect.side, cropRect.side);
    g.fillStyle='rgba(0,0,0,0.35)';
    g.fillRect(cropRect.x, cropRect.y, cropRect.side, 18);
    g.fillStyle='#fff'; g.font='12px system-ui';
    g.fillText(Math.round(cropRect.side)+'×'+Math.round(cropRect.side),
               cropRect.x+4, cropRect.y+13);
    g.restore();
  }
}""", 1)

# 4) 鼠标按下事件
s = s.replace("""cOcc.onclick=e=>clicked(cOcc,e);""",
"""cOcc.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cOcc,e); return;} });
cView.addEventListener('mousedown',e=>{ if(cropMode){e.preventDefault(); cordSel(cView,e); return;} });
cOcc.onclick=e=>clicked(cOcc,e);""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
