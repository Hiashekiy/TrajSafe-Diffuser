"""Isolated 1+8 depth trial. Train from scratch; smoke weights are never reused."""
import argparse,datetime,hashlib,json,os,platform,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run-id',required=True)
    ap.add_argument('--config',default='experiments/safety_baselines/configs/v4_map16_l1_final8.yaml')
    args=ap.parse_args()
    if Path(args.run_id).name!=args.run_id: raise ValueError('run-id must be a basename')
    out=ROOT/'experiments/safety_baselines/results'/args.run_id
    out.mkdir(exist_ok=False)
    cfgpath=args.config
    cfg=load_config(cfgpath);cfg['train']['ckpt_dir']=str(out/'ckpt')
    (out/'config_snapshot.json').write_text(json.dumps(cfg,indent=2),encoding='utf8')
    cmd=[sys.executable,'-u','train.py','--config',cfgpath,'--device','cuda',
         '--ckpt-dir',str(out/'ckpt'),'--out-dir',str(out)]
    manifest={'status':'training','pid':os.getpid(),'python':sys.executable,
      'started_at':datetime.datetime.now(datetime.timezone.utc).isoformat(),
      'command':cmd,'comparison':'v4_map16_l1 1+3 vs 1+8',
      'train_seed':cfg['env']['seed'],'epochs':cfg['train']['epochs'],
      'max_hours':cfg['train']['max_hours'],
      'git_head':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
      'config_sha256':hashlib.sha256((ROOT/cfgpath).read_bytes()).hexdigest()}
    status=out/'manifest.json'
    def save():status.write_text(json.dumps(manifest,indent=2),encoding='utf8')
    save()
    with (out/'train.log').open('w',encoding='utf8') as log:
        rc=subprocess.call(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
    manifest['train_exit_code']=rc
    if rc:
        manifest['status']='training_failed';save();return rc
    summary=json.loads((out/'training_summary.json').read_text())
    if summary['epochs_done']!=cfg['train']['epochs']:
        manifest['status']='training_budget_reached';save();return 2
    manifest['status']='trained';save()
    # Keep test evaluation as a separate explicit step: do not evaluate the smoke model.
    manifest['next_step']='evaluate best_task.pt against original 1+3 on test 420 x seeds 0,1,2; matched-time benchmark'
    save();return 0
if __name__=='__main__':raise SystemExit(main())
