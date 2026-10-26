"""Unified continuous-safety schematic, computed cubic splines and synthetic ALM."""
from pathlib import Path
import json
import numpy as np
from scipy.interpolate import BSpline
from scipy.spatial import ConvexHull
from scipy.optimize import minimize
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, Rectangle, FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D

OUT = Path(__file__).resolve().parent
plt.rcParams.update({'font.family':'Times New Roman','font.size':12,'svg.fonttype':'none','pdf.fonttype':42,'mathtext.fontset':'stix','axes.unicode_minus':False})
BLUE,RED,ORANGE,GRAY,INK,PURPLE = '#3978b7','#c74b4b','#dba75c','#a1a9b2','#26313c','#776a99'
reference = np.array([[0,0],[.7,.1],[1.8,1.5],[3.1,2.0],[4.5,1.2],[5.8,1.0],[7.1,2.0],[8.8,3.0],[10,3.0]])
original = reference.copy()
original[3,1] += 1.25
original[4,1] -= 1.25
original[5,1] -= .85
knots = np.r_[np.zeros(4),np.arange(1,6)/6,np.ones(4)]
regions_t = np.array([0,.22,.48,.72,1])
breaks = np.unique(np.r_[regions_t,knots])
basis = BSpline(knots,np.eye(9),3)

def bezier_map(a,b):
    d = basis.derivative()
    return np.vstack([basis(a),basis(a)+(b-a)/3*d(a),basis(b)-(b-a)/3*d(b),basis(b)])

maps = [bezier_map(a,b) for a,b in zip(breaks,breaks[1:])]
assignments = [min(3,np.searchsorted(regions_t,(a+b)/2,side='right')-1) for a,b in zip(breaks,breaks[1:])]
polygons,halfspaces = [],[]
for region in range(4):
    controls = np.vstack([m@reference for m,r in zip(maps,assignments) if r==region])
    angles = np.arange(12)*2*np.pi/12
    padding = .19*np.column_stack([np.cos(angles),np.sin(angles)])
    cloud = (controls[:,None,:]+padding[None,:,:]).reshape(-1,2)
    hull = ConvexHull(cloud)
    polygons.append(cloud[hull.vertices])
    halfspaces.append(hull.equations)

# Every cubic Bezier control point must satisfy the assigned region halfspaces.
rows,offsets = [],[]
for m,region in zip(maps,assignments):
    for weights in m:
        for nx,ny,b in halfspaces[region]:
            rows.append(np.outer(weights,[nx,ny]).reshape(-1))
            offsets.append(b)
A,b = np.array(rows),np.array(offsets)
fixed = np.r_[0,1,16,17]
free = np.setdiff1d(np.arange(18),fixed)
q0 = original.reshape(-1)
Af = A[:,free]
bf = b+A[:,fixed]@q0[fixed]
raw = q0[free].copy()
x = raw.copy(); lam = np.zeros(len(b)); rho = 5.; trace = []
for iteration in range(70):
    def objective(v):
        g = Af@v+bf
        pos = np.maximum(0,lam+rho*g)
        value = .5*np.sum((v-raw)**2)+np.sum(pos**2-lam**2)/(2*rho)
        grad = v-raw+Af.T@pos
        return value,grad
    result = minimize(objective,x,jac=True,method='L-BFGS-B',options={'maxiter':1000,'ftol':1e-14,'gtol':1e-10,'maxls':50})
    x = result.x; g = Af@x+bf
    violation = max(0,float(np.max(g)))
    lam = np.maximum(0,lam+rho*g)
    trace.append({'iteration':iteration+1,'max_violation':violation,'rho':rho})
    if violation<1e-8: break
    if iteration%4==3: rho = min(rho*2,1e6)
corrected = q0.copy(); corrected[free] = x; corrected = corrected.reshape(-1,2)
assert np.max(A@corrected.reshape(-1)+b)<1e-6
curve0,curve1 = BSpline(knots,original,3),BSpline(knots,corrected,3)
assert np.allclose(curve0([0,1]),curve1([0,1]))
for (a,c),m in zip(zip(breaks,breaks[1:]),maps):
    t = np.linspace(0,1,31)
    bern = np.column_stack([(1-t)**3,3*t*(1-t)**2,3*t*t*(1-t),t**3])
    assert np.max(np.abs(bern@(m@corrected)-curve1(a+(c-a)*t)))<1e-10

fig = plt.figure(figsize=(16,9.5),facecolor='white')
ax = fig.add_axes([.015,.02,.97,.96]); ax.set_xlim(0,160); ax.set_ylim(0,92); ax.axis('off')

def label(x,y,s,size=12,color=INK,ha='center',va='center',**kw):
    return ax.text(x,y,s,fontsize=size,color=color,ha=ha,va=va,**kw)
def arrow(a,b,color=GRAY,lw=1.2,style='-|>',**kw):
    p = FancyArrowPatch(a,b,arrowstyle=style,mutation_scale=11,linewidth=lw,color=color,**kw); ax.add_patch(p); return p
def scene(p):
    p=np.asarray(p);return np.column_stack([12+p[:,0]*13.1,52+p[:,1]*8.2])
def plot_scene(p,**kw):
    q=scene(p);return ax.plot(q[:,0],q[:,1],**kw)

# Compact visual key, no title or panel labels.
ax.plot([8,13],[88,88],color=BLUE,lw=2)
label(14.5,88,'Original B-spline trajectory',13,ha='left')
ax.plot([64,69],[88,88],color=RED,lw=2,ls=(0,(5,3)))
label(70.5,88,'ALM-corrected trajectory',13,ha='left')
ax.plot([118,123],[88,88],color=GRAY,lw=1,marker='o',ms=3.3)
label(124.5,88,'Control polygon',13,ha='left')

for i,p in enumerate(polygons):
    ax.add_patch(Polygon(scene(p),closed=True,facecolor='#f5deb8',edgecolor=ORANGE,alpha=.53,lw=1.1,zorder=1))
    mid=curve1((regions_t[i]+regions_t[i+1])/2)
    ptxt=scene([mid])[0]
    label(ptxt[0],ptxt[1]+(4.2 if i==2 else -6.2),fr'$\mathcal{{C}}_{{{i+1}}}$',14,color='#a37435')
plot_scene(original,color=GRAY,lw=.9,marker='o',ms=4,mfc='#e1e6eb',mec='#8d98a6',zorder=3)
plot_scene(corrected,color=RED,lw=.65,alpha=.3,zorder=3)
ts=np.linspace(0,1,800)
plot_scene(curve0(ts),color=BLUE,lw=2.1,zorder=5)
plot_scene(curve1(ts),color=RED,lw=2.1,ls=(0,(5,3)),zorder=6)
plot_scene(corrected,color=RED,lw=0,marker='o',ms=5.2,mfc='white',mec=RED,mew=1.1,zorder=7)
for j in (3,4,5):
    a,c=scene([original[j],corrected[j]])
    v=c-a; length=np.linalg.norm(v)
    if length>1: arrow(a+v*.16,c-v*.17,color='#b45b59',lw=1.1,zorder=8)
start,goal=scene([original[0],original[-1]])
ax.scatter([start[0],goal[0]],[start[1],goal[1]],s=33,c=INK,zorder=9)
label(start[0]-2.2,start[1]-3.3,'Start',13,ha='right')
label(goal[0]+2.3,goal[1]+2,'Goal',13,ha='left')
label(14,78,'Convex safe corridor',13,ha='left')
arrow((33,76.5),tuple(scene([curve1(.17)])[0]+[-1,2.5]),color='#be975d',lw=.9)
label(51,82.5,'ALM correction',12,color=RED)
arrow((51,80.6),tuple(scene([corrected[3]])[0]+[0,1]),color=RED,lw=.9)
label(86,49,'Original and corrected control points share fixed endpoints',11.5,color='#687482')

# A single progress axis, with responsibility intervals above and refined spans below.
left,right=10,98
xp=lambda t:left+(right-left)*t
label(54,43.3,'Region assignment + knot refinement',14)
for i,(a,c) in enumerate(zip(regions_t,regions_t[1:])):
    ax.add_patch(Rectangle((xp(a),37.8),xp(c)-xp(a),2.9,facecolor='#fff4e3',edgecolor=ORANGE,lw=.9))
    label((xp(a)+xp(c))/2,39.25,fr'Region $\mathcal{{C}}_{{{i+1}}}$',11.5,color='#986a2e')
ax.plot([left,right],[34,34],color=INK,lw=1.1)
arrow((right,34),(right+1.5,34),color=INK,lw=1.1)
for t in regions_t:
    ax.plot([xp(t),xp(t)],[32.1,41],color=ORANGE,lw=1.1,zorder=2)
for t in np.unique(knots)[1:-1]:
    ax.plot([xp(t),xp(t)],[32.7,35.5],color=PURPLE,lw=2)
for t,s in [(0,'0'),(1,'1')]:label(xp(t),30.8,s,11.5)
label(54,35.8,'Trajectory parameter / progress',11.5)
selected=3
for i,(a,c) in enumerate(zip(breaks,breaks[1:])):
    face='#f1e4ca' if i==selected else ('#fbf7ef' if i%2==0 else '#f6f3ed')
    ax.add_patch(Rectangle((xp(a),25.4),xp(c)-xp(a),2.9,facecolor=face,edgecolor='#aa9c85',lw=.7))
    # The narrow interval still denotes a real cubic segment; label it by a leader below.
    if c-a>.035: label((xp(a)+xp(c))/2,26.85,fr'$B_{{{i+1}}}$',11)
    else:
        mid=(xp(a)+xp(c))/2
        ax.plot([mid,mid],[25.4,24.2],color='#8e8578',lw=.7)
        label(mid,23.15,fr'$B_{{{i+1}}}$',10)
label(10,29.4,'Cubic Bézier subsegments',11,ha='left',color='#6c655b')
ax.plot([11,11],[20.1,22.5],color=ORANGE,lw=1.2)
label(13,21.3,'Region boundary',10.8,ha='left')
ax.plot([43,43],[20.4,22.2],color=PURPLE,lw=2)
label(45,21.3,'B-spline knot',10.8,ha='left')
label(75,21.3,'Split at the union of both boundaries',10.5,color='#697581')
arrow((17,48.2),(17,44.8),color='#89949e',lw=1.1)

# Local detail uses the exact segment and its actual assigned region.
ax.add_patch(FancyBboxPatch((106,7),47,34,boxstyle='round,pad=.0,rounding_size=.8',facecolor='white',edgecolor='#aeb8c1',lw=.9))
label(129.5,38,'Exact B-spline → Bézier conversion',12.5)
region=assignments[selected]; polygon=polygons[region]
lo,hi=polygon.min(axis=0),polygon.max(axis=0)
scale=min(36/(hi[0]-lo[0]),15/(hi[1]-lo[1]))
def local(p):
    p=np.asarray(p);return (p-(lo+hi)/2)*scale+np.array([129.5,25.5])
ax.add_patch(Polygon(local(polygon),closed=True,facecolor='#faecd4',edgecolor=ORANGE,lw=1,zorder=1))
bz=maps[selected]@corrected
q=local(bz)
ax.add_patch(Polygon(q,closed=True,fill=False,edgecolor='#9aa4af',ls=(0,(3,2)),lw=.9,zorder=2))
ax.plot(q[:,0],q[:,1],color='#9aa4af',lw=.9,zorder=3)
u=np.linspace(0,1,100)
bern=np.column_stack([(1-u)**3,3*u*(1-u)**2,3*u*u*(1-u),u**3])
lc=local(bern@bz)
ax.plot(lc[:,0],lc[:,1],color=RED,lw=2,ls=(0,(5,2.5)),zorder=4)
ax.scatter(q[:,0],q[:,1],s=24,facecolors='white',edgecolors=RED,linewidths=1,zorder=5)
for i,p in enumerate(q):label(p[0],p[1]+(2 if i%2==0 else -2),fr'$b_{{{i}}}$',11,color=RED)
label(129.5,15.2,'Four Bézier control points in the assigned region',11)
label(129.5,11.4,'Hence, the entire cubic segment stays inside',11.7)
arrow((99.5,27),(105,27),color='#8c7b5d',lw=1.1)
label(102,30,'Zoom',9.5)

# Unified explanatory flow.
blocks=[(10,6,26,'Subdivide and assign\nconvex regions'),(42,6,26,'Constrain Bézier\ncontrol points'),(74,6,27,'ALM correction of\nspline control points')]
for x,y,w,title in blocks:
    ax.add_patch(FancyBboxPatch((x,y),w,9,boxstyle='round,pad=.0,rounding_size=.8',facecolor='#fbfbfa',edgecolor='#aab1b7',lw=.9))
    label(x+w/2,y+4.5,title,11.5)
arrow((36.5,10.5),(41.5,10.5),color='#78848f')
arrow((68.5,10.5),(73.5,10.5),color='#78848f')
arrow((23,24.5),(23,15.5),color='#a4a7a7',lw=.9)

for ext in ('svg','pdf','png'):
    fig.savefig(OUT/f'alm_continuous_safety.{ext}',dpi=300 if ext=='png' else 150,facecolor='white',bbox_inches='tight',pad_inches=.08)
plt.close(fig)
validation={'synthetic_example':True,'degree':3,'spline_control_points':9,'convex_regions':4,'region_boundaries':regions_t.tolist(),'knots':knots.tolist(),'union_partition':breaks.tolist(),'bezier_subsegments':len(maps),'alm_outer_iterations':len(trace),'max_corrected_bezier_halfspace_violation':float(np.max(A@corrected.reshape(-1)+b)),'max_original_bezier_halfspace_violation':float(np.max(A@original.reshape(-1)+b)),'endpoints_fixed':bool(np.allclose(original[[0,-1]],corrected[[0,-1]])),'original_control_points':original.tolist(),'corrected_control_points':corrected.tolist(),'alm_trace':trace,'note':'Computed synthetic ALM illustration; not measured project results. Safety claim is only relative to the assigned illustrative convex corridor.'}
(OUT/'geometry_and_validation.json').write_text(json.dumps(validation,indent=2),encoding='utf-8')
print(json.dumps({k:validation[k] for k in ['bezier_subsegments','alm_outer_iterations','max_corrected_bezier_halfspace_violation','max_original_bezier_halfspace_violation','endpoints_fixed']}))
