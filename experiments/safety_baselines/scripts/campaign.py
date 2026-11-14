"""Staged, resumable nine-method evaluation. Never overwrite prior results."""
import argparse,collections,datetime,hashlib,json,os,platform,statistics,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
import torch,scipy,osqp
from threadpoolctl import threadpool_limits
from src.datasets.carla_spline_dataset import CarlaSplineDataset,make_collate
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample as sample_ours
from src.utils.config import load_config
from src.utils.checkpoint import load_model
from experiments.safety_baselines.models.map_only import load_baseline
from experiments.safety_baselines.adapters.baseline_sampler import sample_baseline
from experiments.safety_baselines.algorithms.runner import run_method,METHODS,derived_seed
from experiments.safety_baselines.adapters.evaluator import evaluate_curve
BASE=ROOT/'experiments/safety_baselines'
OURS_CONFIG='experiments/safety_baselines/configs/v4_map16_l1_final8_e100.yaml'
OURS_CKPT='experiments/safety_baselines/results/depth_l1_final8_e100_20261112_190325/ckpt/best_task.pt'
ALL=METHODS+['Diffuser','TrajSafe-Diffuser']
def digest(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def source_hash():
 files=[p for d in ['algorithms','adapters','models'] for p in (BASE/d).rglob('*.py')]+[Path(__file__),BASE/'scripts/summarize_campaign.py',BASE/'run.py']+list((ROOT/'src').rglob('*.py'))+list((ROOT/'configs').glob('*.yaml'))
 return {str(p.relative_to(ROOT)):digest(p) for p in files}
def dump(path,data):
 temp=path.with_suffix(path.suffix+'.tmp')
 for attempt in range(30):
  try:
   temp.write_text(json.dumps(data,indent=2,allow_nan=False),encoding='utf8');os.replace(temp,path);return
  except PermissionError:
   if attempt==29:raise
   time.sleep(min(.1*(attempt+1),1.))
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['S1','S2','S3'],required=True);ap.add_argument('--run-id',required=True);ap.add_argument('--resume',action='store_true');ap.add_argument('--methods',nargs='+',choices=ALL,default=ALL);ap.add_argument('--previous');ap.add_argument('--config',default='configs/seven_curve_control_t16.json');ap.add_argument('--small-validation',action='store_true');ap.add_argument('--from-small',action='store_true');args=ap.parse_args()
 if args.small_validation and args.stage!='S1':raise ValueError('small validation requires S1')
 if args.from_small and args.stage!='S3':raise ValueError('from-small requires S3')
 direct_full=os.environ.get('TRAJSAFE_DIRECT_FULL')=='1'
 if direct_full and args.stage!='S3':raise ValueError('direct full only supports S3')
 if Path(args.run_id).name!=args.run_id:raise ValueError('invalid run-id')
 config_path=BASE/args.config
 cfg=json.loads(config_path.read_text());protocol=json.loads((BASE/'configs/protocol.json').read_text());stage=protocol[args.stage]
 fingerprint={'config_hash':digest(config_path),'protocol_hash':digest(BASE/'configs/protocol.json'),'source_hashes':source_hash(),'base_hash':digest(ROOT/cfg['checkpoint']),'ours_hash':digest(ROOT/OURS_CKPT)}
 if args.stage!='S1':
  if not args.previous:raise ValueError('previous stage required')
  prev=BASE/'results'/args.previous
  if not prev.resolve().is_relative_to((BASE/'results').resolve()):raise ValueError('invalid previous path')
  old=json.loads((prev/'manifest.json').read_text());expected='S1' if args.stage=='S2' or args.from_small else 'S2'
  if args.from_small and (old.get('split')!='val' or old.get('representation_gate')!='passed'):raise ValueError('small validation integrity gate not passed')
  if direct_full:
   if old['stage']!='S2' or old['status'] not in ('interrupted','complete'):raise ValueError('expected retained S2 results')
   before=old['fingerprint'];changed={k for k in set(before['source_hashes'])|set(fingerprint['source_hashes']) if before['source_hashes'].get(k)!=fingerprint['source_hashes'].get(k)}
   allowed={str(Path(__file__).relative_to(ROOT))}
   if changed-allowed or any(before[k]!=fingerprint[k] for k in fingerprint if k!='source_hashes'):raise ValueError('numerical assets changed; cannot reuse')
  elif old['stage']!=expected or old['status']!='complete' or old['fingerprint']!=fingerprint:raise ValueError('previous stage incomplete or frozen assets changed')
 out=BASE/'results'/args.run_id
 if out.exists() and not args.resume:raise FileExistsError('use explicit --resume')
 out.mkdir(exist_ok=True);records_path=out/'results.jsonl';done={}
 split='val' if args.small_validation else 'test'
 if args.small_validation:
  stage={'indices':cfg['validation_indices'],'seeds':[0]}
 if direct_full and not args.resume:
  retained=[];valid_ids={protocol['scene_ids'][i] for i in stage['indices']};keys=set()
  for line in (prev/'results.jsonl').read_text(encoding='utf8').splitlines():
   r=json.loads(line);key=(r['method'],r['scene_id'],r['seed'])
   if key in keys:raise ValueError('duplicate retained result')
   keys.add(key)
   if r['method'] in args.methods and r['scene_id'] in valid_ids and r['seed'] in stage['seeds']:retained.append(line)
  records_path.write_text('\n'.join(retained)+ ('\n' if retained else ''),encoding='utf8')
  dump(out/'reuse_provenance.json',{'source_run':args.previous,'records_reused':len(retained),'authorization':'user requested direct full evaluation','source_changes':sorted(changed),'note':'only campaign scheduling changed; model, algorithm, evaluator, configuration and per-case seeds identical'})
 if records_path.exists():
  for line in records_path.read_text(encoding='utf8').splitlines():
   r=json.loads(line);key=(r['method'],r['scene_id'],r['seed'])
   if key in done:raise ValueError('duplicate result key')
   done[key]=r
 if args.resume:
  old=json.loads((out/'manifest.json').read_text())
  if old['fingerprint']!=fingerprint or old['methods']!=args.methods:raise ValueError('resume fingerprint differs')
 manifest={'status':'running','stage':args.stage,'pid':os.getpid(),'fingerprint':fingerprint,'methods':args.methods,'seeds':stage['seeds'],'scene_indices':stage['indices'],'expected_records':len(args.methods)*len(stage['indices'])*len(stage['seeds']),'completed_records':len(done),'versions':{'python':platform.python_version(),'torch':torch.__version__,'numpy':np.__version__,'scipy':scipy.__version__,'osqp':osqp.__version__,'shapely':__import__('shapely').__version__},'hardware':torch.cuda.get_device_name(),'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),'randomness':'SHA256(scene_id,seed,method), independent of batch order','preprocessing':'actual map field/projection A* initialization included when used; Ours loads existing map-derived skeleton cache; original offline cache generation cost unavailable and excluded','fidelity':'all external methods task-adapted; local nonconvex projection, approximate score and discrete CBF','performance_gate':'stages check execution integrity and numerical tests, not a required success percentage; failures remain in denominator'}
 manifest.update(split=split,representation_gate='pending' if args.small_validation else 'inherited')
 dump(out/'manifest.json',manifest);dump(out/'config_snapshot.json',cfg)
 model,ck=load_baseline(ROOT/cfg['checkpoint'],'cuda');dc=ck['config']['diffusion'];schedule=NoiseSchedule(dc['timesteps'],dc['beta_schedule'],dc['beta_start'],dc['beta_end']).to('cuda')
 ocfg=load_config(OURS_CONFIG);ours,_,report=load_model(ocfg,OURS_CKPT,device='cuda',verbose=False)
 if any(report.get(k) for k in ['missing','adapted','unexpected']):raise ValueError('Ours checkpoint mismatch')
 ours.eval();ours.assert_shapes=False;ds=CarlaSplineDataset(split,ck['config']['data']['processed_root'],num_controls=48);collate=make_collate(ds)
 scene_ids=[int(ds[i]['sample_id']) for i in range(len(ds))]
 if split=='test' and scene_ids!=protocol['scene_ids']:raise ValueError('test ids changed')
 with torch.inference_mode():
  q=torch.zeros(1,48,2,device='cuda');occ=torch.zeros(1,1,256,256,device='cuda');cond=torch.tensor([[[-.5,0],[.5,0]]],device='cuda');t=torch.zeros(1,dtype=torch.long,device='cuda')
  for _ in range(3):model(q,occ,cond,t)
 torch.cuda.synchronize()
 try:
  with threadpool_limits(limits=1),records_path.open('a',encoding='utf8') as handle:
   for index in stage['indices']:
    for seed in stage['seeds']:
     for method in args.methods:
      scene=scene_ids[index];key=(method,scene,seed)
      if key in done:continue
      started=time.perf_counter();row=ds[index];occ=row['occupancy'][0].numpy();cond=row['cond'].numpy();before=None
      if method in METHODS:
       result=run_method(method,model,schedule,occ,cond,scene,seed,cfg);Q=result['trajectory'];before=result.pop('trajectory_before',None)
      else:
       result={'method':method,'status':'ok','failure_reason':None,'source_id':'map-only conditional diffusion' if method=='Diffuser' else 'full Ours','fidelity':'original project implementation','per_step_trace':[]}
       try:
        s=derived_seed(scene,seed,method)%(2**31-1);batch=collate([row]);o=batch['occupancy'].cuda();c=batch['cond'].cuda()
        if method=='Diffuser':q=sample_baseline(model,schedule,o,c,s,0,16)
        else:
         raw=sample_ours(ours,schedule,c,o,batch['candidate_xy'].cuda(),batch['candidate_mask'].cuda(),batch['candidate_geometry'].cuda(),batch['candidate_geometry_lengths'].cuda(),device='cuda',steps=16,seed=s,return_trace=False,alm_config=ocfg['alm'],corridor_config=ocfg.get('corridor',{}),guidance='alm')
         q=raw['control']
         if not raw['guided'].all():result['status']='failed';result['failure_reason']='corridor_activation_failed'
        Q=q[0].cpu().numpy().tolist()
       except (RuntimeError,ValueError) as exc:Q=None;result['status']='failed';result['failure_reason']=str(exc)
      torch.cuda.synchronize();planning_wall=time.perf_counter()-started
      if planning_wall>cfg.get('timeouts',{}).get(method,cfg['timeout_seconds']):result['status']='failed';result['failure_reason']='timeout'
      metrics=None;common_before=None;representation_error=None
      if Q is not None and result.get('internal_state_trajectory') is not None:
       native=np.asarray(result['internal_state_trajectory']);basis=model.bspline.basis_at(np.linspace(0,1,len(native))).detach().cpu().numpy().astype(float)
       representation_error=float(np.max(np.abs(native-basis@np.asarray(Q))))
       if cfg.get('state_adapter')=='curve_control' and representation_error>1e-10:raise ValueError('optimization/output representation mismatch')
      try:
       if Q is not None:
        metrics=evaluate_curve(model.bspline,Q,occ,cond,row['pos'].numpy(),cfg['endpoint_tol_m'],cfg['radius_scene'])
        if before is not None:
         pre=evaluate_curve(model.bspline,before,occ,cond,row['pos'].numpy(),cfg['endpoint_tol_m'],cfg['radius_scene'])
         union=np.unique(metrics['_parameters']+pre['_parameters'])
         metrics=evaluate_curve(model.bspline,Q,occ,cond,row['pos'].numpy(),cfg['endpoint_tol_m'],cfg['radius_scene'],parameters=union)
         pre=evaluate_curve(model.bspline,before,occ,cond,row['pos'].numpy(),cfg['endpoint_tol_m'],cfg['radius_scene'],parameters=union)
         common_before={k:pre[k] for k in ['common_sdf_violation_max_m','common_sdf_violation_positive_mean_m','common_sdf_min_m']}
        metrics.pop('_parameters',None)
        if result['status']!='ok':metrics['planning_success']=False
      except (ValueError,FloatingPointError,OverflowError) as exc:
       result['status']='failed';result['failure_reason']='numerical_evaluation:'+str(exc);metrics=None
      success=result['status']=='ok' and metrics is not None and metrics['planning_success']
      failure=result['failure_reason']
      if not success and failure is None:failure='collision' if metrics and metrics['collision'] else 'endpoint_error'
      record={**result,'scene_id':scene,'episode_id':int(row['episode_id']),'index':index,'seed':seed,'trajectory':Q,'planning_success':bool(success),'failure_reason':failure,'wall_time':planning_wall,'metrics':metrics,'common_before':common_before,'representation_error_scene':representation_error,'evaluation_wall_time':time.perf_counter()-started-planning_wall}
      handle.write(json.dumps(record,allow_nan=False)+'\n');handle.flush();done[key]=record
      manifest['completed_records']=len(done);manifest['current']={'method':method,'scene_id':scene,'seed':seed};dump(out/'manifest.json',manifest)
      print(args.stage,len(done),'/',manifest['expected_records'],method,scene,seed,success,failure,round(planning_wall,3),flush=True)
  if len(done)!=manifest['expected_records']:raise ValueError('wrong record count')
  if args.small_validation:
   # Validate identity on real outputs; collision percentage is not an integrity gate.
   returned=[r for r in done.values() if r['method'] in METHODS and r['trajectory'] is not None]
   if not returned:raise ValueError('no external method returned a trajectory; investigate before full evaluation')
   if any(r.get('representation_error_scene') is None or r['representation_error_scene']>1e-10 for r in returned):raise ValueError('representation gate failed')
   manifest['representation_gate']='passed'
   missing=set(METHODS)-{r['method'] for r in returned}
   manifest['solver_execution_gate']='failed' if missing else 'passed'
   manifest['methods_without_any_completed_output']=sorted(missing)
   dump(out/'manifest.json',manifest)
   if missing:raise ValueError('small validation: no completed solve for '+', '.join(sorted(missing))+'; full evaluation not launched')
  subprocess.run([sys.executable,str(BASE/'scripts/summarize_campaign.py'),str(out)],cwd=ROOT,check=True)
  manifest['status']='complete';manifest['current']=None;manifest['finished_at']=datetime.datetime.now(datetime.timezone.utc).isoformat();dump(out/'manifest.json',manifest)
 except BaseException as exc:
  manifest['status']='failed';manifest['error']=repr(exc);dump(out/'manifest.json',manifest);raise
if __name__=='__main__':main()
