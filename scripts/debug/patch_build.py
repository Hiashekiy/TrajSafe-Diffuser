"""给 dashboard 加版本戳 + 客户端自动刷新，避免旧进程的旧 JS 静默失败。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

if 'BUILD = time.strftime' not in s:
    s = s.replace("SCENE_DIR = ROOT/'outputs'/'dashboard_scenes'",
                  "BUILD = time.strftime('%Y%m%d_%H%M%S', time.localtime(Path(__file__).stat().st_mtime))\n"
                  "SCENE_DIR = ROOT/'outputs'/'dashboard_scenes'", 1)

if 'const BUILD=' not in s:
    s = s.replace("async function post(u,b){",
                  "const BUILD='__BUILD__';\n"
                  "function checkBuild(s){ if(s&&s.build&&s.build!==BUILD){\n"
                  "  msg('检测到服务端已更新，正在刷新…'); setTimeout(()=>location.reload(true),400); } }\n"
                  "async function post(u,b){", 1)

if 'checkBuild(j)' not in s:
    s = s.replace("    try{ return JSON.parse(txt); }",
                  "    try{ const j=JSON.parse(txt); checkBuild(j); return j; }", 1)

if 'v__BUILD__' not in s:
    s = s.replace('<b style="margin-right:6px">TrajSafe 场景工具</b>',
                  '<b style="margin-right:6px">TrajSafe 场景工具</b>'
                  '<span style="color:#6b7280;font-size:11px;margin-right:8px">v__BUILD__</span>', 1)

if "HTML.replace('__BUILD__'" not in s:
    s = s.replace("""            if self.path in ('/', '/index.html'):
                self._send(200, HTML.encode('utf8'), 'text/html; charset=utf-8')""",
                  """            if self.path in ('/', '/index.html'):
                self._send(200, HTML.replace('__BUILD__', BUILD).encode('utf8'),
                           'text/html; charset=utf-8')""", 1)

if "obj.setdefault('build', BUILD)" not in s:
    s = s.replace("""        def _json(self, obj):
            self._send(200, json.dumps(obj).encode('utf8'), 'application/json; charset=utf-8')""",
                  """        def _json(self, obj):
            obj = dict(obj)
            obj.setdefault('build', BUILD)
            self._send(200, json.dumps(obj).encode('utf8'), 'application/json; charset=utf-8')""", 1)

p.write_text(s, encoding='utf8')
print('patched ok', 'BUILD' in s, '__BUILD__' in s)
