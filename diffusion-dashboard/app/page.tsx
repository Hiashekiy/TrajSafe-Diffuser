'use client';

import { useEffect, useMemo, useState, type MouseEvent as ReactMouseEvent } from 'react';
import { Activity, Ban, Box, ChevronLeft, ChevronRight, CircleDot, Database, Eraser, Flag, Gauge, Layers3, LoaderCircle, Maximize2, MousePointer2, Pause, Play, RotateCcw, Save, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Slider } from '@/components/ui/slider';
import { Switch } from '@/components/ui/switch';
import { Select, SelectContent, SelectItem, SelectTrigger, SelectValue } from '@/components/ui/select';
import rawCatalog from '@/lib/dashboard-catalog.json';

type Vec = [number, number];
type Sample = { key: string; maze: string; datasetId: number; condition: Vec[]; groundTruth: { P: Vec[]; E6: number[][] } };
type Catalog = { provenance: { dataset: string; checkpoints: Record<string, string>; horizon: number; timesteps: number }; maps: Record<string, { resolution: number; wallRuns: number[][] }>; samples: Sample[] };
type Generation = { sampleKey: string; modelId: string; seed: number; cacheHit: boolean; elapsedMs: number; stateLabels: string[]; schedule: { sqrtAlphaBar: number[]; sqrtOneMinusAlphaBar: number[] }; PHistory: Vec[][]; E6History: number[][][]; X0PHistory: Vec[][]; X0E6History: number[][][] };
type EditMode = 'none' | 'start' | 'goal' | 'obstacle' | 'erase';
type DisplayMode = 'state' | 'prediction' | 'compare';
type CustomObstacle = { x: number; y: number; r: number };
const data = rawCatalog as unknown as Catalog;
const API = 'http://localhost:8765';

const models = [
  { id: 'epoch100', name: 'Epoch 100', source: 'center_balanced / epoch_100.pt' },
  { id: 'continue100', name: 'Continue 100 · Best', source: 'continue100 / best.pt' },
  { id: 'continue200', name: 'Continue 200 · Best', source: 'continue200 / best.pt' },
];
const mazeNames: Record<string, string> = { umaze: 'U-Maze', medium: 'Medium', large: 'Large' };
const layerConfig = [
  { key: 'state', label: '主轨迹状态', color: '#50e3ff' }, { key: 'ellipses', label: '对应椭圆状态', color: '#ff6bd6' },
  { key: 'waypoints', label: '128 个 Waypoint', color: '#eef2f5' }, { key: 'result', label: '模型最终输出 P₀', color: '#c7ff4a' },
  { key: 'groundTruth', label: '数据集 Ground Truth', color: '#ffcf5a' }, { key: 'map', label: 'Occupancy Map', color: '#65727e' },
] as const;
const xy = (point: Vec): Vec => [(point[0] + 1) * 128, (1 - point[1]) * 128];
const points = (path: Vec[]) => path.map((point) => xy(point).join(',')).join(' ');
const pathLength = (path: Vec[]) => path.slice(1).reduce((sum, p, i) => sum + Math.hypot(p[0] - path[i][0], p[1] - path[i][1]), 0);

export default function Home() {
  const [sampleKey, setSampleKey] = useState(data.samples[0].key);
  const [modelId, setModelId] = useState(models[0].id);
  const [seed, setSeed] = useState(43);
  const [condition, setCondition] = useState<[Vec, Vec]>([data.samples[0].condition[0], data.samples[0].condition[1]] as [Vec, Vec]);
  const [obstacles, setObstacles] = useState<CustomObstacle[]>([]);
  const [editMode, setEditMode] = useState<EditMode>('none');
  const [displayMode, setDisplayMode] = useState<DisplayMode>('prediction');
  const [history, setHistory] = useState<Generation | null>(null);
  const [status, setStatus] = useState<'idle'|'running'|'ready'|'error'>('idle');
  const [statusText, setStatusText] = useState('选择参数后生成');
  const [backendReady, setBackendReady] = useState(false);
  const [reverseStep, setReverseStep] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const [layers, setLayers] = useState<Record<string, boolean>>({ state: true, ellipses: true, waypoints: true, result: false, groundTruth: false, map: true });

  const sample = data.samples.find((item) => item.key === sampleKey)!;
  const model = models.find((item) => item.id === modelId)!;
  const map = data.maps[sample.maze];
  const maxStep = history ? history.stateLabels.length - 1 : 16;
  const safeStep = Number.isFinite(reverseStep) ? Math.max(0, Math.min(maxStep, reverseStep)) : 0;
  const stateLabel = history?.stateLabels[safeStep] ?? '—';
  const t = stateLabel.startsWith('t=') ? Number(stateLabel.slice(2)) : 0;
  const noisyState = history ? { P: history.PHistory[safeStep], E6: history.E6History[safeStep] } : null;
  const x0Prediction = history?.X0PHistory?.[safeStep] ? { P: history.X0PHistory[safeStep], E6: history.X0E6History[safeStep] } : null;
  const current = displayMode === 'state' ? noisyState : x0Prediction;
  const finalState = history ? { P: history.PHistory.at(-1)!, E6: history.E6History.at(-1)! } : null;

  useEffect(() => { fetch(`${API}/health`).then((response) => { if (response.ok) setBackendReady(true); }).catch(() => setBackendReady(false)); }, []);
  useEffect(() => { if(history && !history.X0PHistory?.length){ setHistory(null); setStatus('idle'); setStatusText('缓存格式已升级，请重新生成'); setReverseStep(0); } }, [history]);
  useEffect(() => { if (!playing || !history) return; const timer = window.setInterval(() => setReverseStep((value) => value >= maxStep ? 0 : value + 1), 700 / speed); return () => window.clearInterval(timer); }, [playing, speed, history, maxStep]);

  const invalidate = () => { setHistory(null); setStatus('idle'); setStatusText('参数已变化，请重新生成'); setReverseStep(0); setPlaying(false); };
  const generate = async () => {
    setStatus('running'); setStatusText('正在运行 16 步联合扩散…'); setPlaying(false);
    try {
      const response = await fetch(`${API}/generate`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ sampleKey, modelId, seed, condition, obstacles: obstacles.map((item)=>[item.x,item.y,item.r]) }) });
      const result = await response.json(); if (!response.ok) throw new Error(result.error || '生成失败');
      setHistory(result); setReverseStep(0); setStatus('ready'); setStatusText(result.cacheHit ? `缓存命中 · ${result.elapsedMs} ms` : `GPU 生成并已缓存 · ${result.elapsedMs} ms`);
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
  const ellipseItems = useMemo(() => !current ? [] : current.E6.map((e,i)=>{ const center:Vec=[current.P[i][0]+e[0],current.P[i][1]+e[1]]; return {center:xy(center),a:Math.min(.42,Math.exp(Math.max(-5,Math.min(-.35,e[2]))))*128,b:Math.min(.42,Math.exp(Math.max(-5,Math.min(-.35,e[3]))))*128,angle:-.5*Math.atan2(e[5],e[4])*180/Math.PI}; }), [current]);

  return <main className="min-h-screen bg-background text-foreground">
    <header className="topbar"><div className="brand-mark"><Sparkles size={18}/></div><div><h1>Diffusion Lens</h1><p>Joint P / E reverse process viewer</p></div><div className={`status-pill ${backendReady?'online':'offline'}`}><span/> {backendReady?'GPU SERVICE READY':'SERVICE OFFLINE'}</div></header>
    <div className="workspace">
      <aside className="left-rail">
        <section><div className="section-label"><Database size={14}/> 数据源</div><div className="source-card"><strong>processed_scene_v1</strong><span>test · 256×256 · H=128</span></div></section>
        <section><label className="field-label">数据集样本</label><Select value={sampleKey} onValueChange={(value)=>selectSample(data.samples.find((item)=>item.key===value)!)}><SelectTrigger className="control-select"><SelectValue/></SelectTrigger><SelectContent>{data.samples.map((item)=><SelectItem key={item.key} value={item.key}>{mazeNames[item.maze]} · #{item.datasetId}</SelectItem>)}</SelectContent></Select><div className="sample-nav"><Button variant="outline" size="icon-sm" aria-label="上一个样本" onClick={()=>chooseRelative(-1)}><ChevronLeft/></Button><span>{sampleIndex+1} / {data.samples.length}</span><Button variant="outline" size="icon-sm" aria-label="下一个样本" onClick={()=>chooseRelative(1)}><ChevronRight/></Button></div></section>
        <section><label className="field-label">模型 checkpoint</label><Select value={modelId} onValueChange={(value)=>{setModelId(value as string);invalidate();}}><SelectTrigger className="control-select model-select"><SelectValue/></SelectTrigger><SelectContent>{models.map((item)=><SelectItem key={item.id} value={item.id}>{item.name}</SelectItem>)}</SelectContent></Select><div className="model-meta"><span>{model.source}</span><span>16 steps</span></div><label className="field-label seed-label">展示状态</label><Select value={displayMode} onValueChange={(value)=>setDisplayMode(value as DisplayMode)}><SelectTrigger className="control-select display-select"><SelectValue/></SelectTrigger><SelectContent><SelectItem value="prediction">每步 x₀ prediction</SelectItem><SelectItem value="state">当前 noisy state Pₜ / Eₜ</SelectItem><SelectItem value="compare">x₀ prediction + noisy state</SelectItem></SelectContent></Select><label className="field-label seed-label" htmlFor="seed">随机种子</label><Input id="seed" className="seed-input" type="number" min={0} max={2147483647} value={seed} onChange={(event)=>{setSeed(Math.max(0,Number(event.target.value)||0));invalidate();}}/><Button className="generate-button" disabled={status==='running'||!backendReady} onClick={generate}>{status==='running'?<LoaderCircle className="spin"/>:<Sparkles/>}{status==='running'?'生成中…':'生成扩散序列'}</Button><div className={`generation-status ${status}`}><Save size={12}/>{statusText}</div></section>
        <section className="layer-section"><div className="section-label"><Layers3 size={14}/> 显示元素</div>{layerConfig.map((item)=><div className="layer-row" key={item.key}><span className="legend-dot" style={{background:item.color}}/><label htmlFor={item.key}>{item.label}</label><Switch id={item.key} checked={layers[item.key]} onCheckedChange={(checked)=>setLayers((old)=>({...old,[item.key]:checked}))}/></div>)}</section>
      </aside>

      <section className="stage-column">
        <div className="stage-head"><div><span className="eyebrow">{displayMode==='state'?'NOISY STATE':displayMode==='prediction'?'X₀ PREDICTION':'PREDICTION / STATE'} · {stateLabel}</span><h2>{mazeNames[sample.maze]} / dataset #{sample.datasetId}</h2></div><Button variant="ghost" size="icon" aria-label="全屏查看" onClick={()=>document.querySelector('.canvas-shell')?.requestFullscreen()}><Maximize2/></Button></div>
        <div className="edit-toolbar"><span><MousePointer2 size={13}/> 点击编辑</span><button className={editMode==='start'?'active':''} onClick={()=>setEditMode(editMode==='start'?'none':'start')}><CircleDot/>起点</button><button className={editMode==='goal'?'active':''} onClick={()=>setEditMode(editMode==='goal'?'none':'goal')}><Flag/>终点</button><button className={editMode==='obstacle'?'active':''} onClick={()=>setEditMode(editMode==='obstacle'?'none':'obstacle')}><Ban/>添加障碍</button><button className={editMode==='erase'?'active':''} onClick={()=>setEditMode(editMode==='erase'?'none':'erase')}><Eraser/>擦除障碍</button><button onClick={resetEdits}><RotateCcw/>重置</button><em>{obstacles.length} 个自定义障碍</em></div>
        <div className={`canvas-shell ${editMode!=='none'?'editing':''}`}><svg viewBox="0 0 256 256" role="img" aria-label="真实占据地图上的联合轨迹与椭圆扩散状态" onClick={handleCanvasClick}><rect width="256" height="256" fill="#d8dde1"/>{layers.map&&<g>{map.wallRuns.map(([x,y,w],i)=><rect key={i} x={x} y={y} width={w} height="1.15" fill="#10161b" opacity=".96"/>)}</g>}<g>{obstacles.map((item,i)=>{const q=xy([item.x,item.y]);return <circle key={i} cx={q[0]} cy={q[1]} r={item.r*128} fill="#10161b" stroke="#ff6b72" strokeWidth=".8"/>;})}</g>{layers.groundTruth&&<polyline points={points(sample.groundTruth.P)} fill="none" stroke="#ffcf5a" strokeWidth="1.15" strokeDasharray="3 2" opacity=".9"/>}{layers.result&&finalState&&<polyline points={points(finalState.P)} fill="none" stroke="#c7ff4a" strokeWidth="1.25" opacity=".9"/>}{displayMode==='compare'&&layers.state&&noisyState&&<polyline points={points(noisyState.P)} fill="none" stroke="#50e3ff" strokeWidth=".8" strokeLinejoin="round" opacity=".48"/>}{layers.ellipses&&<g>{ellipseItems.filter((_,i)=>i%8===0).map((e,i)=><ellipse key={i} cx={e.center[0]} cy={e.center[1]} rx={e.a} ry={e.b} transform={`rotate(${e.angle} ${e.center[0]} ${e.center[1]})`} fill="#ff6bd608" stroke="#ff6bd6" strokeWidth=".7" opacity=".7"/>)}</g>}{layers.state&&current&&<polyline points={points(current.P)} fill="none" stroke={displayMode==='state'?'#50e3ff':'#c7ff4a'} strokeWidth={displayMode==='compare'?'1.6':'1.35'} strokeLinejoin="round" opacity=".92"/>}{layers.waypoints&&current&&<g>{current.P.map((p,i)=>{const q=xy(p);return <circle key={i} cx={q[0]} cy={q[1]} r="1.05" fill={displayMode==='state'?'#eaf7fa':'#edffc7'} opacity=".82"/>;})}</g>}<g><circle cx={xy(condition[0])[0]} cy={xy(condition[0])[1]} r="3.4" fill="#55f17a" stroke="#071009" strokeWidth="1"/><path d={`M${xy(condition[1])[0]-4},${xy(condition[1])[1]}h8M${xy(condition[1])[0]},${xy(condition[1])[1]-4}v8`} stroke="#ff5d67" strokeWidth="2"/></g></svg>{!history&&editMode==='none'&&<div className="generation-empty"><Sparkles/><strong>等待生成</strong><span>可先点击编辑起终点与障碍<br/>再运行一次模型</span><Button disabled={!backendReady} onClick={generate}>生成扩散序列</Button></div>}<div className="canvas-badge">{editMode!=='none'?`编辑：${editMode==='start'?'起点':editMode==='goal'?'终点':editMode==='obstacle'?'添加障碍':'擦除障碍'}`:history?(stateLabel==='x0'?'final x₀':displayMode==='state'?<>x<sub>{t}</sub> state</>:<>x̂<sub>0</sub><sup>({t})</sup></>):'no sequence'}</div><div className="axis-label x">scene x ∈ [-1, 1]</div><div className="axis-label y">scene y ∈ [-1, 1]</div></div>
        <div className="transport"><div className="transport-main"><Button variant="outline" size="icon" disabled={!history} aria-label="返回纯噪声" onClick={()=>{setReverseStep(0);setPlaying(false);}}><RotateCcw/></Button><Button className="play-button" size="icon-lg" disabled={!history} aria-label={playing?'暂停':'播放'} onClick={()=>setPlaying(!playing)}>{playing?<Pause/>:<Play/>}</Button><div className="timeline"><Slider value={[safeStep]} max={maxStep} step={1} disabled={!history} onValueChange={(value)=>{const next=Array.isArray(value)?value[0]:value;setReverseStep(Number(next)||0);setPlaying(false);}} aria-label="反向扩散进度"/><div><span>t=15 · Gaussian noise</span><strong>{history?`${safeStep}/${maxStep}`:'—'}</strong><span>final x₀ · model output</span></div></div><div className="speed-control">{[.25,.5,1,2,4].map((item)=><button key={item} className={speed===item?'active':''} onClick={()=>setSpeed(item)}>{item}×</button>)}</div></div></div>
        <p className="method-note">“x₀ prediction”展示每个 timestep 模型刚输出的 P̂₀⁽ᵗ⁾ / Ê₀⁽ᵗ⁾；“noisy state”展示送入模型的 Pₜ / Eₜ。两套张量都会随生成结果缓存。</p>
      </section>

      <aside className="right-rail"><div className="metric-title"><Activity size={15}/> 当前扩散状态</div><div className="hero-metric"><span>{displayMode==='state'?'noisy state':displayMode==='prediction'?'x₀ prediction':'prediction + state'}</span><strong>{stateLabel}<small>{history&&stateLabel==='x0'?' · final':history?' / 15':''}</small></strong><div className="meter"><i style={{width:`${history?safeStep/maxStep*100:0}%`}}/></div></div><div className="metric-grid"><div><span>√ᾱₜ</span><strong>{history?(stateLabel==='x0'?1:history.schedule.sqrtAlphaBar[t]).toFixed(4):'—'}</strong></div><div><span>√(1−ᾱₜ)</span><strong>{history?(stateLabel==='x0'?0:history.schedule.sqrtOneMinusAlphaBar[t]).toFixed(4):'—'}</strong></div><div><span>{displayMode==='state'?'Pₜ → P₀ RMSE':'P̂₀⁽ᵗ⁾ → P₀ RMSE'}</span><strong>{rmse===null?'—':rmse.toFixed(3)}</strong></div><div><span>碰撞 / 越界点</span><strong>{collisionCount===null?'—':`${collisionCount} / 128`}</strong></div><div><span>当前路径长度</span><strong>{current?pathLength(current.P).toFixed(2):'—'}</strong></div><div><span>缓存状态</span><strong>{history?(history.cacheHit?'HIT':'SAVED'):'—'}</strong></div></div><div className="step-log"><div className="metric-title"><Gauge size={15}/> 任务状态</div><div className="phase"><i/>{statusText}</div><p>{history?(displayMode==='state'?'当前显示 sampler 输入的 noisy state。':'当前显示模型在该 timestep 直接预测的 clean x₀，可用于判断轨迹何时成型。'):'模型尚未运行；地图与 Ground Truth 可先用于确认样本。'}</p></div><div className="provenance"><Box size={14}/><div><strong>推理来源</strong><span>{data.provenance.dataset}<br/>{data.provenance.checkpoints[modelId]}</span></div></div></aside>
    </div>
  </main>;
}
