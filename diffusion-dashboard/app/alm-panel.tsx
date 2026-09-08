'use client';

import { useEffect, useMemo, useState } from 'react';
import { Activity, ArrowLeftRight, Box, Gauge, LoaderCircle, Pause, Play, RotateCcw, Sparkles } from 'lucide-react';
import { Button } from '@/components/ui/button';
import { Slider } from '@/components/ui/slider';
import { Tabs, TabsList, TabsTrigger } from '@/components/ui/tabs';

type Vec = [number, number];
type Obstacle = { x: number; y: number; r: number };
type MapData = { resolution: number; wallRuns: number[][] };
type Trace = {
  sampleKey: string;
  rawTrajectory: Vec[];
  finalTrajectory: Vec[];
  rawCenters: Vec[];
  repairedCenters: Vec[];
  ellipseShape: number[][];
  regionValid: boolean[];
  regions: Vec[][];
  trajectoryFrames: { inner: number; P: Vec[] }[];
  stats: Record<string, number>;
};

export type LensView = 'diffusion' | 'alm';

export function ViewTabs({ value, onChange }: { value: LensView; onChange: (value: LensView) => void }) {
  return <Tabs value={value} onValueChange={(next) => onChange(next as LensView)} className="lens-tabs">
    <TabsList><TabsTrigger value="diffusion">扩散过程</TabsTrigger><TabsTrigger value="alm">ALM 修正</TabsTrigger></TabsList>
  </Tabs>;
}

const xy = (point: Vec): Vec => [(point[0] + 1) * 128, (1 - point[1]) * 128];
const linePoints = (path: Vec[]) => path.map((point) => xy(point).join(',')).join(' ');
const polygonPoints = (polygon: Vec[]) => polygon.map((point) => xy(point).join(',')).join(' ');

function EllipseMark({ center, shape, repaired, dim = false }: { center: Vec; shape: number[]; repaired: boolean; dim?: boolean }) {
  const q = xy(center);
  const rx = Math.min(.42, Math.exp(Math.max(-5, Math.min(-.35, shape[0])))) * 128;
  const ry = Math.min(.42, Math.exp(Math.max(-5, Math.min(-.35, shape[1])))) * 128;
  const angle = -.5 * Math.atan2(shape[3], shape[2]) * 180 / Math.PI;
  return <ellipse cx={q[0]} cy={q[1]} rx={rx} ry={ry} transform={`rotate(${angle} ${q[0]} ${q[1]})`}
    fill={repaired ? '#00e5ff18' : 'none'} stroke={repaired ? '#00e5ff' : '#ff3fbf'}
    strokeWidth={repaired ? 1.4 : 1} strokeDasharray={repaired ? undefined : '4 2'} opacity={dim ? .3 : 1}/>;
}

export function ALMPanel({
  onViewChange, history, sampleKey, sampleName, datasetId, modelName, map,
  condition, obstacles, backendReady,
}: {
  onViewChange: (value: LensView) => void;
  history: { X0PHistory: Vec[][]; X0E6History: number[][][] } | null;
  sampleKey: string;
  sampleName: string;
  datasetId: number;
  modelName: string;
  map: MapData;
  condition: [Vec, Vec];
  obstacles: Obstacle[];
  backendReady: boolean;
}) {
  const [trace, setTrace] = useState<Trace | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [frame, setFrame] = useState(0);
  const [playing, setPlaying] = useState(false);
  const [speed, setSpeed] = useState(1);
  const centerFrames = trace?.rawCenters.length ?? 128;
  const totalFrames = trace ? centerFrames + trace.trajectoryFrames.length : 133;
  const safeFrame = Math.max(0, Math.min(totalFrames - 1, frame));
  const centerPhase = safeFrame < centerFrames;
  const centerIndex = Math.min(centerFrames - 1, safeFrame);
  const trajectoryIndex = Math.max(0, safeFrame - centerFrames);
  const trajectory = trace?.trajectoryFrames[trajectoryIndex]?.P ?? trace?.rawTrajectory;

  useEffect(() => { setTrace(null); setFrame(0); setPlaying(false); }, [sampleKey, modelName, history]);
  useEffect(() => {
    if (!playing || !trace) return;
    const timer = window.setInterval(() => setFrame((old) => old >= totalFrames - 1 ? 0 : old + 1), 650 / speed);
    return () => window.clearInterval(timer);
  }, [playing, speed, trace, totalFrames]);
  useEffect(() => {
    const onKey = (event: KeyboardEvent) => {
      if (!trace || ['INPUT', 'SELECT', 'TEXTAREA'].includes((event.target as HTMLElement)?.tagName)) return;
      if (event.key === 'ArrowRight') { event.preventDefault(); setPlaying(false); setFrame((old) => Math.min(totalFrames - 1, old + 1)); }
      if (event.key === 'ArrowLeft') { event.preventDefault(); setPlaying(false); setFrame((old) => Math.max(0, old - 1)); }
      if (event.key === ' ') { event.preventDefault(); setPlaying((old) => !old); }
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [trace, totalFrames]);

  const generateTrace = async () => {
    if (!history) return;
    setLoading(true); setError(''); setPlaying(false);
    try {
      const response = await fetch('http://localhost:8765/alm-trace', {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ sampleKey, condition, obstacles: obstacles.map((item) => [item.x, item.y, item.r]), rawP: history.X0PHistory.at(-1), rawE6: history.X0E6History.at(-1) }),
      });
      const result = await response.json();
      if (!response.ok) throw new Error(result.error || 'ALM trace 生成失败');
      setTrace(result); setFrame(0);
    } catch (reason) { setError(reason instanceof Error ? reason.message : '无法连接 ALM 服务'); }
    finally { setLoading(false); }
  };

  const movedCount = useMemo(() => !trace ? 0 : trace.rawCenters.reduce((sum, center, index) =>
    sum + (Math.hypot(center[0] - trace.repairedCenters[index][0], center[1] - trace.repairedCenters[index][1]) > 1e-6 ? 1 : 0), 0), [trace]);
  const activeRegionIndex = useMemo(() => {
    if (!trace || !centerPhase) return -1;
    for (let index = centerIndex; index >= 0; index--) if (trace.regionValid[index] && trace.regions[index].length >= 3) return index;
    return -1;
  }, [trace, centerPhase, centerIndex]);
  const activeRegionCenter = useMemo(() => {
    if (!trace || activeRegionIndex < 0) return null;
    const polygon = trace.regions[activeRegionIndex];
    return xy([polygon.reduce((sum, p) => sum + p[0], 0) / polygon.length, polygon.reduce((sum, p) => sum + p[1], 0) / polygon.length]);
  }, [trace, activeRegionIndex]);
  const phaseLabel = centerPhase ? `中心递推 · ellipse ${centerIndex + 1}/${centerFrames}` : `轨迹 ALM · inner ${trace?.trajectoryFrames[trajectoryIndex]?.inner ?? 0}`;

  return <main className="min-h-screen bg-background text-foreground">
    <header className="topbar"><div className="brand-mark"><Sparkles size={18}/></div><div><h1>Diffusion Lens</h1><p>Joint P / E reverse process viewer</p></div><ViewTabs value="alm" onChange={onViewChange}/><div className={`status-pill ${backendReady ? 'online' : 'offline'}`}><span/> {backendReady ? 'GPU SERVICE READY' : 'SERVICE OFFLINE'}</div></header>
    <div className="alm-workspace">
      <aside className="alm-side">
        <div className="section-label"><Box size={14}/> 当前输入</div>
        <div className="source-card"><strong>{sampleName} · #{datasetId}</strong><span>{modelName}<br/>final x₀ prediction · H=128</span></div>
        <p className="alm-help">先在“扩散过程”生成序列，再用最终 x₀ 构造完整的中心传播与轨迹 ALM 回放。</p>
        <Button className="generate-button" disabled={!history || loading || !backendReady} onClick={generateTrace}>{loading ? <LoaderCircle className="spin"/> : <Sparkles/>}{loading ? '生成 ALM 序列…' : trace ? '重新生成 ALM 序列' : '生成 ALM 修正序列'}</Button>
        {!history && <div className="alm-warning">尚无扩散结果，请先切回扩散过程生成。</div>}{error && <div className="alm-warning error">{error}</div>}
        <div className="alm-legend"><div><i className="raw"/>原始椭圆（洋红虚线）</div><div><i className="fixed"/>修正椭圆（青色实线）</div><div><i className="region"/>当前凸区域（蓝色块）</div><div><i className="path-before"/>ALM 前轨迹（洋红虚线）</div><div><i className="path-after"/>当前修正轨迹（青色实线）</div></div>
        <div className="keyboard-note"><ArrowLeftRight size={14}/><span><kbd>←</kbd><kbd>→</kbd> 单步　<kbd>Space</kbd> 播放/暂停</span></div>
      </aside>

      <section className="alm-stage">
        <div className="stage-head"><div><span className="eyebrow">ALM CORRECTION TRACE</span><h2>{sampleName} / dataset #{datasetId}</h2></div><span className="alm-phase-pill">{trace ? phaseLabel : '等待生成'}</span></div>
        <div className="alm-canvas"><svg viewBox="0 0 256 256" role="img" aria-label="椭圆中心递推与轨迹 ALM 修正过程">
          <defs><filter id="region-glow" x="-30%" y="-30%" width="160%" height="160%"><feGaussianBlur stdDeviation="1.3" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter></defs>
          <rect width="256" height="256" fill="#aeb7bd"/><g>{map.wallRuns.map(([x,y,w],i)=><rect key={i} x={x} y={y} width={w} height="1.15" fill="#0c1217" opacity=".98"/>)}</g>
          <g>{obstacles.map((item,i)=>{const q=xy([item.x,item.y]);return <circle key={i} cx={q[0]} cy={q[1]} r={item.r*128} fill="#10161b" stroke="#ff6b72" strokeWidth=".8"/>;})}</g>
          {trace && centerPhase && <>
            <polyline points={linePoints(trace.rawTrajectory)} fill="none" stroke="#6c7a84" strokeWidth=".65" opacity=".45"/>
            {activeRegionIndex >= 0 && <g filter="url(#region-glow)"><polygon points={polygonPoints(trace.regions[activeRegionIndex])} fill="#075cff70" stroke="#00f5ff" strokeWidth="2.2"/><polygon points={polygonPoints(trace.regions[activeRegionIndex])} fill="none" stroke="#061a2b" strokeWidth="3.8" opacity=".6"/><polygon points={polygonPoints(trace.regions[activeRegionIndex])} fill="none" stroke="#00f5ff" strokeWidth="1.8"/>{activeRegionCenter && <text x={activeRegionCenter[0]} y={activeRegionCenter[1]} textAnchor="middle" dominantBaseline="middle" fill="#ffffff" stroke="#041019" strokeWidth="3" paintOrder="stroke" fontSize="7" fontWeight="800">{activeRegionIndex === centerIndex ? `C${centerIndex}` : `R${activeRegionIndex}`}</text>}</g>}
            <g>{trace.rawCenters.map((center,index)=>(index % 4 === 0 || index === centerIndex) && <EllipseMark key={`r${index}`} center={center} shape={trace.ellipseShape[index]} repaired={false} dim={index !== centerIndex}/>)}</g>
            <g>{trace.repairedCenters.slice(0, centerIndex + 1).map((center,index)=>(index % 4 === 0 || index === centerIndex) && <EllipseMark key={`f${index}`} center={center} shape={trace.ellipseShape[index]} repaired dim={index !== centerIndex}/>)}</g>
            <circle cx={xy(trace.rawCenters[centerIndex])[0]} cy={xy(trace.rawCenters[centerIndex])[1]} r="2.5" fill="#ff3fbf" stroke="#240018" strokeWidth=".8"/>
            <circle cx={xy(trace.repairedCenters[centerIndex])[0]} cy={xy(trace.repairedCenters[centerIndex])[1]} r="2.5" fill="#00e5ff" stroke="#001b22" strokeWidth=".8"/>
            <line x1={xy(trace.rawCenters[centerIndex])[0]} y1={xy(trace.rawCenters[centerIndex])[1]} x2={xy(trace.repairedCenters[centerIndex])[0]} y2={xy(trace.repairedCenters[centerIndex])[1]} stroke="#fff" strokeWidth=".75" strokeDasharray="2 2"/>
          </>}
          {trace && !centerPhase && <>
            <g>{trace.repairedCenters.filter((_,index)=>index%8===0).map((center,index)=><EllipseMark key={index} center={center} shape={trace.ellipseShape[index*8]} repaired dim/>)}</g>
            <polyline points={linePoints(trace.rawTrajectory)} fill="none" stroke="#ff3fbf" strokeWidth="1.6" strokeDasharray="5 2.5" opacity="1"/>
            {trajectory && <polyline points={linePoints(trajectory)} fill="none" stroke="#00e5ff" strokeWidth="2.1" strokeLinejoin="round"/>}
          </>}
          <circle cx={xy(condition[0])[0]} cy={xy(condition[0])[1]} r="3.4" fill="#55f17a" stroke="#071009" strokeWidth="1"/><path d={`M${xy(condition[1])[0]-4},${xy(condition[1])[1]}h8M${xy(condition[1])[0]},${xy(condition[1])[1]-4}v8`} stroke="#ff5d67" strokeWidth="2"/>
        </svg>{!trace && <div className="generation-empty"><Activity/><strong>等待 ALM trace</strong><span>{history ? '生成后可逐椭圆查看中心传播与凸区域' : '请先生成扩散序列'}</span></div>}</div>
        <div className="transport alm-transport"><div className="transport-main"><Button variant="outline" size="icon" disabled={!trace} onClick={()=>{setFrame(0);setPlaying(false);}}><RotateCcw/></Button><Button className="play-button" size="icon-lg" disabled={!trace} onClick={()=>setPlaying(!playing)}>{playing?<Pause/>:<Play/>}</Button><div className="timeline"><Slider value={[safeFrame]} max={totalFrames-1} step={1} disabled={!trace} onValueChange={(value)=>{setFrame(Number(value[0])||0);setPlaying(false);}}/><div><span>ellipse 1 · start anchor</span><strong>{trace?`${safeFrame+1}/${totalFrames}`:'—'}</strong><span>trajectory · ALM final</span></div></div><div className="speed-control">{[.25,.5,1,2,4].map((item)=><button key={item} className={speed===item?'active':''} onClick={()=>setSpeed(item)}>{item}×</button>)}</div></div></div>
      </section>

      <aside className="alm-metrics"><div className="metric-title"><Gauge size={15}/> 修正诊断</div><div className="hero-metric"><span>当前阶段</span><strong className="alm-stage-number">{centerPhase ? centerIndex + 1 : `i${trace?.trajectoryFrames[trajectoryIndex]?.inner ?? 0}`}</strong><div className="meter"><i style={{width:`${trace ? safeFrame/(totalFrames-1)*100 : 0}%`}}/></div></div><div className="metric-grid alm-grid"><div><span>实际移动中心</span><strong>{trace?`${movedCount} / 128`:'—'}</strong></div><div><span>当前区域</span><strong>{trace&&centerPhase?(trace.regionValid[centerIndex]?'VALID':'INVALID'):'—'}</strong></div><div><span>相邻区域重叠率</span><strong>{trace?`${(trace.stats.adjacent_region_overlap_rate*100).toFixed(1)}%`:'—'}</strong></div><div><span>修正后中心不安全率</span><strong>{trace?`${(trace.stats.center_post_unsafe_rate*100).toFixed(2)}%`:'—'}</strong></div><div><span>碰撞覆盖率</span><strong>{trace?`${(trace.stats.collision_covered_rate*100).toFixed(1)}%`:'—'}</strong></div><div><span>碰撞率前 → 后</span><strong>{trace?`${(trace.stats.physical_collision_rate_before*100).toFixed(2)} → ${(trace.stats.physical_collision_rate_after*100).toFixed(2)}%`:'—'}</strong></div></div><div className="step-log"><div className="metric-title"><Activity size={15}/> 当前帧</div><div className="phase"><i/>{phaseLabel}</div><p>{centerPhase?'洋红虚线为原始椭圆，亮青实线为修正椭圆；深蓝色块与粗青边框标出当前 Cₖ。':'洋红虚线为 ALM 前轨迹，亮青实线为当前 inner iteration 的轨迹。'}</p></div></aside>
    </div>
  </main>;
}
