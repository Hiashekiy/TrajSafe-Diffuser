"""修 st.region 丢失导致点击被静默忽略。"""
from pathlib import Path

p = Path('scripts/dashboard_app.py')
s = p.read_text(encoding='utf8')

s = s.replace("function apply(s){\n  st=s;", "function apply(s){\n  st=Object.assign({},st,s);", 1)

s = s.replace("""async function clicked(canvas,ev){
  if(!st.region){return;}
""", """async function clicked(canvas,ev){
""", 1)

s = s.replace("""                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           msg=status()))
                if self.path == '/api/save':""",
"""                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=status()))
                if self.path == '/api/save':""", 1)

s = s.replace("""                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'], msg=note))""",
"""                    return self._json(dict(ok=True, occ=occ_png(w), view=view_png(w),
                                           ego=STATE['ego'], goal=STATE['goal'],
                                           region=STATE['region'], msg=note))""", 1)

p.write_text(s, encoding='utf8')
print('patched ok')
