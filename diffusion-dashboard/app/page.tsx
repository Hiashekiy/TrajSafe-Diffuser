'use client';

import { useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { Activity, Ban, Box, ChevronLeft, ChevronRight, CircleDot, Database, Eraser, Flag, Gauge, Layers3, LoaderCircle, Maximize2, MousePointer2, Pause, Play, RotateCcw, Save, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import rawCatalog from '@/lib/dashboard-catalog-carla.json';

type Vec = [number, number];
type Sample = { key: string; maze: string; datasetId: number; condition: Vec[]; groundTruth: { P: Vec[]; control?: Vec[] }; quality?: { curve_rmse_m: number; collision: boolean } };
type Catalog = { provenance: { dataset: string; checkpoints: Record<string, string>; horizon: number; timesteps: number }; maps: Record<string, { resolution: number; wallRuns: number[][] }>; samples: Sample[] };
type AlmRegion = { i: number; polygon: Vec[] };
type CorridorCell = { i: number; source: string; anchor_s: number; center: Vec; face_count: number; valid: boolean; polygon: Vec[] };
type BridgeGap = { gap: number; anchor_s: number; center: Vec; overlap_left: number; overlap_right: number; overlap_before: number };
type Corridor = { valid: boolean; failure_reason: string | null; min_overlap: number; base_cell_count: number; bridge_cell_count: number; num_cells: number; overlap_ratio: number[]; bridge_gaps: BridgeGap[]; cells: CorridorCell[] };
type ActivationInfo = { step: number; frozen_topology_idx: number; status: string; guided: boolean; warmup_reverse_steps: number; max_activation_delay_steps: number; attempts: number[]; topology_fallback: boolean[]; failure_reason: (string | null)[]; candidate_trials: number[]; region_face_counts: number[]; overlap_min: number | null; overlap_mean: number | null };
type FinalValidation = { final_collision: boolean; final_free_rate: number; endpoint_error: number; dense_points: number; final_max_constraint_violation: number | null; final_constraint_feasible: boolean | null; final_corridor_membership_rate: number | null };
type PackSummary = { num_pieces: number[]; num_pieces_max: number; num_faces_max: number; num_controls: number; num_active_constraints: number };
type AlmFrame = { t: number; raw_p: Vec[]; safe_p?: Vec[]; guided?: boolean; alm_active?: boolean; enforced: number[]; regions: AlmRegion[]; stats: Record<string, number> };
type AlmInfo = { enabled: boolean; mode: string; start_t: number; warmup_reverse_steps: number; activation_step: number; frozen_topology_idx: number; status: string; rho: number; constraint_tol: number; max_curve_step_scene: number; frames: (AlmFrame | null)[] };
type TopologyFrame = { t: number; selected_idx: number; pi: number[]; regions: AlmRegion[]; progress: number[] | null; center: Vec[]; shape4: number[][] };
type TopologyInfo = { epoch: number | null; selection: string; num_candidates: number; num_valid: number; raw_k: number; selected_idx: number; pi: number[]; topology_path: Vec[]; candidate_lengths: number[]; node_path: number[]; frames: (TopologyFrame | null)[]; metrics: { traj_collision: boolean; raw_traj_collision?: boolean; ellipse_count: number; ellipse_collision_rate: number | null; center_free_rate: number | null; min_center_clearance_cells: number | null; region_count: number; progress_monotonic: boolean; step_jitter?: number; selection_changes?: number; alm_status?: string; activation_step?: number; frozen_topology_idx?: number }; skeleton: { nodes: number; branches: number }; candidate_paths?: Vec[][]; candidate_mask?: boolean[]; geometry_points?: number };
type EllipseFrame = { center: Vec[]; shape4: number[][] };
type ControlFrame = Vec[];
type Generation = { sample_key: string; model_id: string; seed: number; cache_hit: boolean; elapsed_ms: number; state_labels: string[]; schedule: { sqrt_alpha_bar: number[]; sqrt_one_minus_alpha_bar: number[] }; state_history: Vec[][]; x0_history: Vec[][]; control_history?: ControlFrame[]; ellipse_history: EllipseFrame[]; alm: AlmInfo | null; corridor?: Corridor | null; activation?: ActivationInfo | null; pack_summary?: PackSummary | null; final_validation?: FinalValidation | null; x0_raw_history?: Vec[][]; topology?: TopologyInfo | null; error?: string };
type EditMode = 'none' | 'start' | 'goal' | 'obstacle' | 'erase';
type DisplayMode = 'state' | 'prediction' | 'compare';
type CustomObstacle = { x: number; y: number; r: number };
const data = rawCatalog as unknown as Catalog;
const API = 'http://localhost:8765';

const models = [
  { id: 'best_task', name: 'best_task · ep449', source: 'ckpt/best_task.pt · 任务指标最优' },
  { id: 'best', name: 'best · ep420', source: 'ckpt/best.pt · val 最优' },
  { id: 'latest', name: 'latest · ep469', source: 'ckpt/latest.pt · 训练末' },
  { id: 'best_run1', name: 'run1 · ep184', source: 'ckpt/best_run1.pt · 第一段' },
];
const mazeNames: Record<string, string> = { umaze: 'U-Maze', medium: 'Medium', large: 'Large' };
const mazeLabel = (key: string): string => mazeNames[key] ?? key.replace('carla_', 'CARLA #');
const layerConfig = [
  { key: 'state', label: '主轨迹状态', color: '#50e3ff' }, { key: 'ellipses', label: '对应椭圆状态', color: '#ff6bd6' },
  { key: 'waypoints', label: '128 个 Waypoint', color: '#eef2f5' }, { key: 'controls', label: '预测的 32 控制点', color: '#ff9f1c' }, { key: 'gtControls', label: 'GT 的 32 控制点（固定）', color: '#c08a2e' }, { key: 'result', label: '模型最终输出 P₀', color: '#c7ff4a' },
  { key: 'groundTruth', label: '数据集 Ground Truth', color: '#ffcf5a' }, { key: 'map', label: 'Occupancy Map', color: '#65727e' },
  { key: 'rawX0', label: 'raw x̂₀（修正前，洋红）/ ALM safe x̂₀（青色）', color: '#ff5ebf' }, { key: 'regions', label: '冻结安全走廊 · 128 基础区域（cyan）/ bridge（橙色）', color: '#2fb9ff' },
  { key: 'topology', label: '候选搜索路径 + 选中拓扑', color: '#4aa3ff' },
] as const;
const xy = (point: Vec): Vec => [(point[0] + 1) * 128, (1 - point[1]) * 128];
const points = (path: Vec[]) => path.map((point) => xy(point).join(',')).join(' ');
const pathLength = (path: Vec[]) => path.slice(1).reduce((sum, p, i) => sum + Math.hypot(p[0] - path[i][0], p[1] - path[i][1]), 0);
// 椭圆颜色沿轨迹顺序由浅到深：idx 越小（越靠近起点）越浅，越大（越靠近终点）越深。
const ellipseColor = (idx: number, total: number): { stroke: string; fill: string } => {
  const t = total > 1 ? Math.max(0, Math.min(1, idx / (total - 1))) : 0;
  const h = 300 + 15 * t;          // hue 300 -> 315 (粉 -> 紫)
  const s = 60 + 25 * t;           // saturation 60 -> 85
  const l = 90 - 52 * t;           // lightness 90 -> 38（浅 -> 深）
  const css = `hsl(${h} ${s}% ${l}%)`;
  return { stroke: css, fill: css };
};

export default function Home() {
  const [sampleKey, setSampleKey] = useState(data.samples[0].key);
  const [modelId, setModelId] = useState(models[0].id);
  const [seed, setSeed] = useState(43);
  const [condition, setCondition] = useState<[Vec, Vec]>([data.samples[0].condition[0], data.samples[0].condition[1]] as [Vec, Vec]);
  const [obstacles, setObstacles] = useState<CustomObstacle[]>([]);
  const [editMode, setEditMode] = useState<EditMode>('none');
  const [displayMode, setDisplayMode] = useState<DisplayMode>('prediction');
  const [almEnabled, setAlmEnabled] = useState(false);
  const [history, setHistory] = useState<Generation | null>(null);
  const [status, setStatus] = useState<'idle'|'running'|'ready'|'error'>('idle');
  const [statusText, setStatusText] = useState('选择参数后生成');
  const [backendReady, setBackendReady] = useState(false);
  const [backendEngine, setBackendEngine] = useState('');
  const [reverseStep, setReverseStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [layers, setLayers] = useState<Record<string, boolean>>({ state: true, ellipses: true, waypoints: true, controls: true, gtControls: false, result: true, groundTruth: true, map: true, rawX0: true, regions: true, topology: true });

  const sample = data.samples.find((item) => item.key === sampleKey)!;
  const model = models.find((item) => item.id === modelId)!;
  const map = data.maps[sample.maze];
  const maxStep = history ? history.state_labels.length - 1 : 16;
  const safeStep = Number.isFinite(reverseStep) ? Math.max(0, Math.min(maxStep, reverseStep)) : 0;
  const stateLabel = history?.state_labels[safeStep] ?? '—';
  const t = stateLabel.startsWith('t=') ? Number(stateLabel.slice(2)) : 0;
  const noisyState = history ? { P: history.state_history[safeStep] } : null;
  const x0Prediction = history?.x0_history?.[safeStep] ? { P: history.x0_history[safeStep] } : null;
  const current = displayMode === 'state' ? noisyState : x0Prediction;
  const finalState = history ? { P: history.state_history.at(-1)! } : null;
  const ellipseFrame = history?.ellipse_history?.[safeStep] ?? null;
  const controlPts = history?.control_history?.[safeStep] ?? null;
  const editedCondition = Math.abs(condition[0][0]-sample.condition[0][0])>1e-6 || Math.abs(condition[0][1]-sample.condition[0][1])>1e-6 || Math.abs(condition[1][0]-sample.condition[1][0])>1e-6 || Math.abs(condition[1][1]-sample.condition[1][1])>1e-6;
  const almFrame = history?.alm?.frames?.[safeStep] ?? null;
  const almVisible = !!almFrame && displayMode !== 'state';
  const corridor = history?.corridor ?? null;
  const corridorCells = corridor?.cells ?? null;
  const corridorBridgeGaps = corridor?.bridge_gaps ?? [];
  const activation = history?.activation ?? null;
  const finalValidation = history?.final_validation ?? null;
  const packSummary = history?.pack_summary ?? null;
  const topologyInfo = history?.topology ?? null;
  const topologyFrame = topologyInfo?.frames?.[safeStep] ?? null;

  useEffect(() => { fetch(`${API}/health`).then((r) => r.ok ? r.json() : Promise.reject(new Error(String(r.status)))).then((info) => { setBackendReady(true); setBackendEngine(String(info?.engine ?? 'legacy')); }).catch(() => { setBackendReady(false); setBackendEngine(''); }); }, []);
  const engineMismatch = backendReady && backendEngine !== 'carla-controlspace-32';
  useEffect(() => { if(history && (!history.x0_history?.length || history.alm === undefined)){ setHistory(null); setStatus('idle'); setStatusText('缓存格式已升级，请重新生成'); setReverseStep(0); } else if(history && history.control_history === undefined){ setStatusText('提示：后端未提供控制点层（旧缓存/旧进程），重启 backend.py 后重新生成即可看到橙色控制点'); } }, [history]);
  useEffect(() => { if (!playing || !history) return; const timer = window.setInterval(() => setReverseStep((value) => value >= maxStep ? 0 : value + 1), 700 / speed); return () => window.clearInterval(timer); }, [playing, speed, history, maxStep]);

  const invalidate = () => { setHistory(null); setStatus('idle'); setStatusText('参数已变化，请重新生成'); setReverseStep(0); setPlaying(false); };
  const generate = async () => {
    setStatus('running'); setStatusText('正在运行 16 步联合扩散…'); setPlaying(false);
    try {
      const response = await fetch(`${API}/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sample_key: sampleKey, model_id: modelId, seed, condition, obstacles: obstacles.map((item)=>[item.x,item.y,item.r]), alm_enabled: almEnabled }) });
      const result = (await response.json()) as Generation; if (!response.ok) throw new Error(result.error || '生成失败');
      setHistory(result); setReverseStep(Math.max(0, (result.state_labels?.length ?? 1) - 1)); setStatus('ready'); setStatusText(result.cache_hit ? `缓存命中 · ${result.elapsed_ms} ms` : `GPU 生成并已缓存 · ${result.elapsed_ms} ms`);
    } catch (error) { setStatus('error'); setStatusText(error instanceof Error ? error.message : '无法连接生成服务'); }
  };

  const sampleIndex = data.samples.findIndex((item) => item.key === sampleKey);
  const selectSample = (nextSample: Sample) => { setSampleKey(nextSample.key); setCondition([nextSample.condition[0],nextSample.condition[1]] as [Vec,Vec]); setObstacles([]); invalidate(); };
  const chooseRelative = (delta: number) => { const next = (sampleIndex + delta + data.samples.length) % data.samples.length; selectSample(data.samples[next]); };
  const handleCanvasClick = (event: ReactMouseEvent<SVGSVGElement>) => {
    if(editMode==='none'||status==='running')return;
    const rect=event.currentTarget.getBoundingClientRect();
    const point:Vec=[Math.max(-1,Math.min(1,(event.clientX-rect.left)/rect.width*2-1)),Math.max(-1,Math.min(1,1-(event.clientY-rect.top)/rect.height*2))];
    if(editMode==='start')setCondition((old)=>[point,old[1]]);
    if(editMode==='goal')setCondition((old)=>[old[0],point]);
    if(editMode==='obstacle')setObstacles((old)=>[...old,{x:point[0],y:point[1],r:.055}]);
    if(editMode==='erase')setObstacles((old)=>old.filter((item)=>Math.hypot(item.x-point[0],item.y-point[1])>item.r+.035));
    invalidate();
  };
  const resetEdits=()=>{setCondition([sample.condition[0],sample.condition[1]] as [Vec,Vec]);setObstacles([]);setEditMode('none');invalidate();};
  const collisionCount = current ? current.P.filter((p) => { const [px,py]=xy(p); if(px<0||px>=256||py<0||py>=256)return true; const x=Math.floor(px),y=Math.floor(py); return map.wallRuns.some(([rx,ry,w])=>ry===y&&x>=rx&&x<rx+w); }).length : null;
  const rmse = current && finalState ? Math.sqrt(current.P.reduce((sum,p,i)=>sum+(p[0]-finalState.P[i][0])**2+(p[1]-finalState.P[i][1])**2,0)/256) : null;
  const ellipseItems = useMemo(() => !ellipseFrame ? [] : ellipseFrame.center.map((c,i)=>{ const s4=ellipseFrame.shape4[i]; return {idx:i,center:xy(c),a:Math.min(.42,Math.exp(Math.max(-5,Math.min(-.35,s4[0]))))*128,b:Math.min(.42,Math.exp(Math.max(-5,Math.min(-.35,s4[1]))))*128,angle:-.5*Math.atan2(s4[3],s4[2])*180/Math.PI}; }), [ellipseFrame]);

  return <main className="min-h-screen bg-background text-foreground">
    <header className="topbar"><div className="brand-mark"><Sparkles size={18}/></div><div><h1>Diffusion Lens</h1><p>{'TrajSafe-Diffuser · CARLA + 32-control B-spline 控制点扩散（每步重选骨架）'}</p></div><div className={`status-pill ${backendReady && !engineMismatch ?'online':'offline'}`} title={engineMismatch?'当前 8765 端口跑的是旧 Maze2D 推理服务：请关掉那个窗口，重新运行 diffusion-dashboard\\backend.py，然后刷新页面':''}><span/> {!backendReady?'SERVICE OFFLINE':engineMismatch?'WRONG BACKEND (legacy Maze2D) — 请重启 backend.py':'GPU SERVICE READY'}</div></header>
    <div className="workspace">
      <aside className="left-rail">
        <section><div className="section-label"><Database size={14}/> 数据源</div><div className="source-card"><strong>carla_processed/test</strong><span>CARLA 80 m crop · 256×256 · H=128 · 32 controls</span></div></section>
        <section><label className="field-label">数据集样本</label><Select value={sampleKey} onValueChange={(value)=>selectSample(data.samples.find((item)=>item.key===value)!)}><SelectTrigger className="control-select"><SelectValue/></SelectTrigger><SelectContent alignItemWithTrigger={false} sideOffset={4}>{data.samples.map((item)=><SelectItem key={item.key} value={item.key}>#{item.datasetId}{item.quality ? ` · ${item.quality.curve_rmse_m.toFixed(2)} m` : ''}</SelectItem>)}</SelectContent></Select><div className="sample-nav"><Button variant="outline" size="icon-sm" aria-label="上一个样本" onClick={()=>chooseRelative(-1)}><ChevronLeft/></Button><span>{sampleIndex+1} / {data.samples.length}</span><Button variant="outline" size="icon-sm" aria-label="下一个样本" onClick={()=>chooseRelative(1)}><ChevronRight/></Button></div></section>
        <section><label className="field-label">模型 checkpoint</label><Select value={modelId} onValueChange={(value)=>{setModelId(value as string);invalidate();}}><SelectTrigger className="control-select model-select"><SelectValue/></SelectTrigger><SelectContent alignItemWithTrigger={false} sideOffset={4}>{models.map((item)=><SelectItem key={item.id} value={item.id}>{item.name}</SelectItem>)}</SelectContent></Select><div className="model-meta"><span>{model.source}</span><span>16 steps</span></div><label className="field-label seed-label">展示状态</label><Select value={displayMode} onValueChange={(value)=>setDisplayMode(value as DisplayMode)}><SelectTrigger className="control-select display-select"><SelectValue/></SelectTrigger><SelectContent><SelectItem value="prediction">每步 x₀ prediction</SelectItem><SelectItem value="state">当前 noisy state Pₜ</SelectItem><SelectItem value="compare">x₀ prediction + noisy state</SelectItem></SelectContent></Select><div className="layer-row"><span className="legend-dot" style={{background:'#c7ff4a'}}/><label htmlFor="almEnabled">安全走廊 + B-spline ALM 引导（16 步中冻结走廊后逐步修正）</label><Switch id="almEnabled" checked={almEnabled} onCheckedChange={(checked)=>{setAlmEnabled(checked);invalidate();}}/></div><label className="field-label seed-label" htmlFor="seed">随机种子</label><Input id="seed" className="seed-input" type="number" min={0} max={2147483647} value={seed} onChange={(event)=>{setSeed(Math.max(0,Number(event.target.value)||0));invalidate();}}/><Button className="generate-button" disabled={status==='running'||!backendReady} onClick={generate}>{status==='running'?<LoaderCircle className="spin"/>:<Sparkles/>}{status==='running'?'生成中…':'生成扩散序列'}</Button><div className={`generation-status ${status}`}><Save size={12}/>{statusText}</div></section>
        <section className="layer-section"><div className="section-label"><Layers3 size={14}/> 显示元素</div>{layerConfig.map((item)=><div className="layer-row" key={item.key}><span className="legend-dot" style={{background:item.color}}/><label htmlFor={item.key}>{item.label}</label><Switch id={item.key} checked={layers[item.key]} onCheckedChange={(checked)=>setLayers((old)=>({...old,[item.key]:checked}))}/></div>)}</section>
      </aside>

      <section className="stage-column">
        <div className="stage-head"><div><span className="eyebrow">{displayMode==='state'?'NOISY STATE':displayMode==='prediction'?'X₀ PREDICTION':'PREDICTION / STATE'} · {stateLabel}{almVisible?` · coarse t=${almFrame.t}`:''}{topologyInfo?` · argmax m=${topologyInfo.selected_idx} · jitter=${topologyInfo.metrics.step_jitter ?? 0}`:''}</span><h2>{mazeLabel(sample.maze)} / dataset #{sample.datasetId}{sample.quality && !editedCondition ? ` · GT 对比误差 ${sample.quality.curve_rmse_m.toFixed(2)} m` : ''}{editedCondition ? ' · 起终点已修改：黄色 GT 属原始样本，不可直接对比' : ''}</h2></div><Button variant="ghost" size="icon" aria-label="全屏查看" onClick={()=>document.querySelector('.canvas-shell')?.requestFullscreen()}><Maximize2/></Button></div>
        <div className="edit-toolbar"><span><MousePointer2 size={13}/> 点击编辑</span><button className={editMode==='start'?'active':''} onClick={()=>setEditMode(editMode==='start'?'none':'start')}><CircleDot/>起点</button><button className={editMode==='goal'?'active':''} onClick={()=>setEditMode(editMode==='goal'?'none':'goal')}><Flag/>终点</button><button className={editMode==='obstacle'?'active':''} onClick={()=>setEditMode(editMode==='obstacle'?'none':'obstacle')}><Ban/>添加障碍</button><button className={editMode==='erase'?'active':''} onClick={()=>setEditMode(editMode==='erase'?'none':'erase')}><Eraser/>擦除障碍</button><button onClick={resetEdits}><RotateCcw/>重置</button><em>{obstacles.length} 个自定义障碍</em></div>
        <div className={`canvas-shell ${editMode!=='none'?'editing':''}`}><svg viewBox="0 0 256 256" role="img" aria-label="真实占据地图上的轨迹与椭圆扩散状态" onClick={handleCanvasClick}><rect width="256" height="256" fill="#d8dde1"/>{layers.map&&<g>{map.wallRuns.map(([x,y,w],i)=><rect key={i} x={x} y={y} width={w} height="1.15" fill="#10161b" opacity=".96"/>)}</g>}<g>{obstacles.map((item,i)=>{const q=xy([item.x,item.y]);return <circle key={i} cx={q[0]} cy={q[1]} r={item.r*128} fill="#10161b" stroke="#ff6b72" strokeWidth=".8"/>;})}</g>{layers.groundTruth&&<polyline points={points(sample.groundTruth.P)} fill="none" stroke="#ffcf5a" strokeWidth="1.15" strokeDasharray="3 2" opacity={editedCondition?'.28':'.9'}/>}{layers.result&&finalState&&<polyline points={points(finalState.P)} fill="none" stroke="#c7ff4a" strokeWidth="1.25" opacity=".9"/>}{displayMode==='compare'&&layers.state&&noisyState&&<polyline points={points(noisyState.P)} fill="none" stroke="#50e3ff" strokeWidth=".8" strokeLinejoin="round" opacity=".48"/>}{topologyInfo&&layers.topology&&<g>{topologyInfo.candidate_paths?.map((path, i)=><polyline key={`cand-${i}`} points={points(path)} fill="none" stroke="#4aa3ff" strokeWidth=".6" strokeDasharray="1.5 2.2" strokeLinejoin="round" opacity={i===topologyInfo.selected_idx?0.0:(topologyInfo.candidate_mask?.[i]===false?0.12:0.3)}/>)}<polyline points={points(topologyInfo.topology_path)} fill="none" stroke="#4aa3ff" strokeWidth="1.15" strokeDasharray="2.5 2" strokeLinejoin="round" opacity=".95"/></g>}{displayMode!=='state'&&layers.regions&&corridorCells&&<g>{corridorCells.map((r)=>{const q=r.polygon.map(xy);const isBridge=r.source==='bridge';return <polygon key={r.i} points={q.map((p)=>p.join(',')).join(' ')} fill={isBridge?'#ffd166':'#25c8ff'} fillOpacity={isBridge?'.30':'.07'} stroke={isBridge?'#ff9f1c':'#2fb9ff'} strokeWidth={isBridge?'.9':'.5'} strokeOpacity=".9"/>;})}{corridorBridgeGaps.map((g)=>{const c=xy(g.center);return <g key={'bg'+g.gap}><circle cx={c[0]} cy={c[1]} r="2.6" fill="#ff9f1c" stroke="#10161b" strokeWidth=".5"/><text x={c[0]} y={c[1]-3.6} fontSize="5" textAnchor="middle" fill="#ff9f1c">bridge</text></g>;})}</g>}{almFrame&&displayMode!=='state'&&layers.rawX0&&<g>{almFrame.raw_p&&<polyline points={points(almFrame.raw_p)} fill="none" stroke="#ff5ebf" strokeWidth="1.05" strokeDasharray="4 2.4" strokeLinejoin="round" opacity=".9"/>}{almFrame.safe_p&&almFrame.alm_active&&<polyline points={points(almFrame.safe_p)} fill="none" stroke="#25c8ff" strokeWidth="1.25" strokeLinejoin="round" opacity=".95"/>}</g>}{layers.ellipses&&<g>{ellipseItems.filter((e)=>e.idx%8===0).map((e)=>{const c=ellipseColor(e.idx,current?.P.length ?? 128);return <ellipse key={e.idx} cx={e.center[0]} cy={e.center[1]} rx={e.a} ry={e.b} transform={`rotate(${e.angle} ${e.center[0]} ${e.center[1]})`} fill={c.fill} fillOpacity=".09" stroke={c.stroke} strokeWidth=".75" opacity=".92"/>;})}</g>}{layers.gtControls&&sample.groundTruth.control&&<g><polyline points={points(sample.groundTruth.control)} fill="none" stroke="#c08a2e" strokeWidth=".7" strokeDasharray="2.4 1.6" opacity=".75"/>{sample.groundTruth.control.map((p,i)=>{const q=xy(p);return <circle key={'gc'+i} cx={q[0]} cy={q[1]} r=".8" fill="#c08a2e" opacity=".75"/>;})}</g>}{layers.controls&&<g>{controlPts&&<polyline points={points(controlPts)} fill="none" stroke="#ff9f1c" strokeWidth=".8" strokeDasharray="2 1.4" opacity=".95"/>}{controlPts&&controlPts.map((p,i)=>{const q=xy(p);return <circle key={'c'+i} cx={q[0]} cy={q[1]} r={i===0||i===controlPts.length-1?'1.7':'1.0'} fill={i===0?'#2ecc71':i===controlPts.length-1?'#ff5b5b':'#ff9f1c'} stroke="#0b1015" strokeWidth=".25"/>;})}</g>}{layers.state&&current&&<polyline points={points(current.P)} fill="none" stroke={displayMode==='state'?'#50e3ff':'#c7ff4a'} strokeWidth={displayMode==='compare'?'1.6':'1.35'} strokeLinejoin="round" opacity=".92"/>}{layers.waypoints&&current&&<g>{current.P.map((p,i)=>{const q=xy(p);return <circle key={i} cx={q[0]} cy={q[1]} r="1.05" fill={displayMode==='state'?'#eaf7fa':'#edffc7'} opacity=".82"/>;})}</g>}<g><circle cx={xy(condition[0])[0]} cy={xy(condition[0])[1]} r="3.4" fill="#55f17a" stroke="#071009" strokeWidth="1"/><path d={`M${xy(condition[1])[0]-4},${xy(condition[1])[1]}h8M${xy(condition[1])[0]},${xy(condition[1])[1]-4}v8`} stroke="#ff5d67" strokeWidth="2"/></g></svg>{!history&&editMode==='none'&&<div className="generation-empty"><Sparkles/><strong>等待生成</strong><span>可先点击编辑起终点与障碍<br/>再运行一次模型</span><Button disabled={!backendReady} onClick={generate}>生成扩散序列</Button></div>}<div className="canvas-badge">{editMode!=='none'?`编辑：${editMode==='start'?'起点':editMode==='goal'?'终点':editMode==='obstacle'?'添加障碍':'擦除障碍'}`:history?(stateLabel==='x0'?'final x₀':displayMode==='state'?<>x<sub>{t}</sub> state</>:<>x̂<sub>0</sub><sup>({t})</sup></>):'no sequence'}</div><div className="axis-label x">scene x ∈ [-1, 1]</div><div className="axis-label y">scene y ∈ [-1, 1]</div></div>
        <div className="transport"><div className="transport-main"><Button variant="outline" size="icon" disabled={!history} aria-label="返回纯噪声" onClick={()=>{setReverseStep(0);setPlaying(false);}}><RotateCcw/></Button><Button className="play-button" size="icon-lg" disabled={!history} aria-label={playing?'暂停':'播放'} onClick={()=>setPlaying(!playing)}>{playing?<Pause/>:<Play/>}</Button><div className="timeline"><Slider value={[safeStep]} max={maxStep} step={1} disabled={!history} onValueChange={(value)=>{const next=Array.isArray(value)?value[0]:value;setReverseStep(Number(next)||0);setPlaying(false);}} aria-label="反向扩散进度"/><div><span>t=15 · Gaussian noise</span><strong>{history?`${safeStep}/${maxStep}`:'—'}</strong><span>final x₀ · model output</span></div></div><div className="speed-control">{[.25,.5,1,2,4].map((item)=><button key={item} className={speed===item?'active':''} onClick={()=>setSpeed(item)}>{item}×</button>)}</div></div></div>
        <p className="method-note">“x₀ prediction”展示每个 timestep 模型刚输出的 P̂₀⁽ᵗ⁾ / Ê₀⁽ᵗ⁾；“noisy state”展示送入模型的 Pₜ / Eₜ。每步重选骨架候选：浅蓝虚线是当前 occupancy/起终点下的搜索路径，深蓝实线是选中的拓扑。开启凸区域修正时，t≤start_t 的帧会叠加预测椭圆生成的 verified convex region（青色）和修正前 x̂₀（粉色虚线）；关闭时粉色虚线为主干 coarse x̂₀。椭圆颜色由浅到深表示它们沿轨迹的顺序（起点→终点）。</p>
      </section>

      <aside className="right-rail"><div className="metric-title"><Activity size={15}/> 当前扩散状态</div><div className="hero-metric"><span>{displayMode==='state'?'noisy state':displayMode==='prediction'?'x₀ prediction':'prediction + state'}</span><strong>{stateLabel}<small>{history&&stateLabel==='x0'?' · final':history?' / 15':''}</small></strong><div className="meter"><i style={{width:`${history?safeStep/maxStep*100:0}%`}}/></div></div><div className="metric-grid"><div><span>√ᾱₜ</span><strong>{history?(stateLabel==='x0'?1:history.schedule.sqrt_alpha_bar[t]).toFixed(4):'—'}</strong></div><div><span>√(1−ᾱₜ)</span><strong>{history?(stateLabel==='x0'?0:history.schedule.sqrt_one_minus_alpha_bar[t]).toFixed(4):'—'}</strong></div><div><span>{displayMode==='state'?'Pₜ → P₀ RMSE':'P̂₀⁽ᵗ⁾ → P₀ RMSE'}</span><strong>{rmse===null?'—':rmse.toFixed(3)}</strong></div><div><span>碰撞 / 越界点</span><strong>{collisionCount===null?'—':`${collisionCount} / 128`}</strong></div><div><span>当前路径长度</span><strong>{current?pathLength(current.P).toFixed(2):'—'}</strong></div><div><span>缓存状态</span><strong>{history?(history.cache_hit?'HIT':'SAVED'):'—'}</strong></div></div><div className="step-log"><div className="metric-title"><Gauge size={15}/> {topologyFrame?'安全诊断':almFrame?'ALM 修正诊断':'任务状态'}</div>{topologyInfo&&topologyFrame&&<div className="metric-grid"><div><span>每步重选 m</span><strong>每步 / {topologyFrame.selected_idx}</strong></div><div><span>π(m)</span><strong>{topologyFrame.pi[topologyFrame.selected_idx]?.toFixed(3) ?? '—'}</strong></div><div><span>候选数 valid / total</span><strong>{topologyInfo.num_valid} / {topologyInfo.num_candidates}</strong></div><div><span>中心安全率 CenterFree</span><strong>{topologyInfo.metrics.center_free_rate===null?'—':(topologyInfo.metrics.center_free_rate*100).toFixed(2)+'%'}</strong></div><div><span>中心最小余量</span><strong>{topologyInfo.metrics.min_center_clearance_cells===null?'—':topologyInfo.metrics.min_center_clearance_cells.toFixed(2)+' 格'}</strong></div><div><span>轨迹碰撞</span><strong>{topologyInfo.metrics.traj_collision?'有':'无'}</strong></div><div><span>椭圆点碰撞率</span><strong>{topologyInfo.metrics.ellipse_collision_rate===null?'—':(topologyInfo.metrics.ellipse_collision_rate*100).toFixed(2)+'%'}</strong></div><div><span>本帧验证凸区域</span><strong>{topologyFrame.regions?.length ?? 0}</strong></div><div><span>进度单调</span><strong>{topologyInfo.metrics.progress_monotonic?'是':'否'}</strong></div><div><span>骨架 nodes / branches</span><strong>{topologyInfo.skeleton.nodes} / {topologyInfo.skeleton.branches}</strong></div></div>}{almFrame&&almFrame.alm_active&&almFrame.stats&&almFrame.stats.max_violation_before!==undefined&&<div className="metric-grid"><div><span>走廊违约 before → after</span><strong>{almFrame.stats.max_violation_before.toFixed(4)} → {almFrame.stats.max_violation_after.toFixed(4)}</strong></div><div><span>平均正违约 before → after</span><strong>{almFrame.stats.mean_positive_violation_before.toFixed(4)} → {almFrame.stats.mean_positive_violation_after.toFixed(4)}</strong></div><div><span>约束可行率</span><strong>{(almFrame.stats.constraint_feasible_rate*100).toFixed(2)}%</strong></div><div><span>有效不等式数</span><strong>{almFrame.stats.active_constraint_count.toFixed(0)}</strong></div><div><span>曲线修正 mean / max (m)</span><strong>{almFrame.stats.mean_curve_correction_m.toFixed(3)} / {almFrame.stats.max_curve_correction_m.toFixed(3)}</strong></div><div><span>λ mean / max</span><strong>{almFrame.stats.lambda_mean.toFixed(3)} / {almFrame.stats.lambda_max.toFixed(3)}</strong></div><div><span>本步 ALM 迭代</span><strong>{almFrame.stats.inner_steps_used.toFixed(0)}</strong></div><div><span>该帧</span><strong>{almFrame.guided?'GUIDED（Q0_safe 进入 DDIM）':'WARMUP（raw）'}</strong></div></div>}{activation&&<div className="metric-grid"><div><span>激活 reverse step</span><strong>{activation.step<0?'未激活':activation.step}</strong></div><div><span>状态</span><strong>{activation.status}</strong></div><div><span>冻结 topology m</span><strong>{activation.guided?activation.frozen_topology_idx:'—'}</strong></div><div><span>候选尝试 / fallback</span><strong>{(activation.attempts[0]??'—')}{activation.topology_fallback[0]?' · fallback':''}</strong></div><div><span>基础区域 / bridge</span><strong>{corridor?`${corridor.base_cell_count} / ${corridor.bridge_cell_count}`:'—'}</strong></div><div><span>overlap min / mean</span><strong>{corridor&&corridor.overlap_ratio.length?`${Math.min(...corridor.overlap_ratio).toFixed(3)} / ${(corridor.overlap_ratio.reduce((a,b)=>a+b,0)/corridor.overlap_ratio.length).toFixed(3)}`:'—'}</strong></div><div><span>约束 piece / 有效面</span><strong>{packSummary?`${packSummary.num_pieces[0]} / ${packSummary.num_active_constraints}`:'—'}</strong></div><div><span>走廊状态</span><strong>{corridor?(corridor.valid?'有效（已冻结）':`失败：${corridor.failure_reason??''}`):'未建立'}</strong></div></div>}{finalValidation&&<div className="metric-grid"><div><span>稠密验证点数</span><strong>{finalValidation.dense_points}</strong></div><div><span>最终碰撞</span><strong>{finalValidation.final_collision?'有':'无'}</strong></div><div><span>终点误差</span><strong>{finalValidation.endpoint_error.toExponential(2)}</strong></div><div><span>最终最大违约</span><strong>{finalValidation.final_max_constraint_violation===null?'—':finalValidation.final_max_constraint_violation.toFixed(5)}</strong></div><div><span>走廊归属率</span><strong>{finalValidation.final_corridor_membership_rate===null?'—':(finalValidation.final_corridor_membership_rate*100).toFixed(1)+'%'}</strong></div><div><span>raw / safe 轨迹碰撞</span><strong>{topologyInfo?.metrics.raw_traj_collision?'raw 有':'raw 无'} / {topologyInfo?.metrics.traj_collision?'safe 有':'safe 无'}</strong></div></div>}<div className="phase"><i/>{statusText}</div><p>{history?(displayMode==='state'?'当前显示 sampler 输入的 noisy state。':almFrame&&almFrame.alm_active?'亮青实线=ALM 修正后的 x̂₀（真正进入 DDIM），洋红虚线=网络原始 x̂₀；青色为冻结走廊的 128 个基础凸区域，橙色高亮为其上的 gap bridge 区域。':'该帧处于 WARMUP：尚未建立走廊，DDIM 直接使用网络原始 x̂₀。'):'模型尚未运行；地图与 Ground Truth 可先用于确认样本。'}</p></div><div className="provenance"><Box size={14}/><div><strong>推理来源</strong><span>{data.provenance.dataset}<br/>{data.provenance.checkpoints[modelId]}</span></div></div></aside>
    </div>
  </main>;
}
