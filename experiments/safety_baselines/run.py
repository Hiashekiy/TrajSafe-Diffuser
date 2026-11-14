"""Unified S0/S1/S2/S3 entry. Use the project's CUDA Python environment."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--stage',choices=['S0','S1','S2','S3'],default='S0');p.add_argument('--run-id');p.add_argument('--previous');p.add_argument('--methods',nargs='+');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 plan={'stage':a.stage,'scope':'seven task-adapted methods plus Diffuser and full Ours','base':'map_only_l1_final8_e100','ours':'v4_map16_l1_final8_e100','run_id':a.run_id,'previous':a.previous,'required_gate':'unit tests' if a.stage=='S1' else 'S1 complete with identical assets' if a.stage=='S2' else 'S2 complete with identical assets' if a.stage=='S3' else None}
 print(json.dumps(plan,indent=2),flush=True)
 if a.dry_run:return 0
 env=dict(os.environ);env.update(PYTHONDONTWRITEBYTECODE='1',OPENBLAS_NUM_THREADS='1',OMP_NUM_THREADS='1');env['PYTHONPATH']=os.pathsep.join([str(ROOT),env.get('PYTHONPATH','')])
 if a.stage=='S0':cmd=[sys.executable,'-m','pytest','-q','-p','no:cacheprovider',str(HERE/'tests'),'--basetemp',str(HERE/'results'/('s0_tmp_'+str(time.time_ns())))]
 else:
  if not a.run_id:p.error('--run-id required')
  cmd=[sys.executable,'-u',str(HERE/'scripts/campaign.py'),'--stage',a.stage,'--run-id',a.run_id]
  if a.previous:cmd+=['--previous',a.previous]
  if a.resume:cmd+=['--resume']
  if a.methods:cmd+=['--methods',*a.methods]
 return subprocess.call(cmd,cwd=ROOT,env=env)
if __name__=='__main__':raise SystemExit(main())
