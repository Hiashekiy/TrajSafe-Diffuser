"""Frozen, resumable paper campaign. Never overwrite historical runs or labels."""
import argparse
import datetime
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import traceback

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from src.utils.config import load_config

def write(path, data):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(data,indent=2,ensure_ascii=False),encoding='utf-8')
    tmp.replace(path)

def prepare(out):
    import yaml
    snap=out/'source_snapshot'
    if snap.exists():
        raise RuntimeError('Snapshot exists; use --resume to preserve experiment provenance')
    out.mkdir(parents=True,exist_ok=True)
    for directory in ('src','scripts'):
        shutil.copytree(ROOT/directory,snap/directory,ignore=shutil.ignore_patterns('__pycache__','*.pyc','node_modules','.next','dist'))
    (snap/'diffusion-dashboard').mkdir()
    for path in (ROOT/'diffusion-dashboard').glob('*.py'):
        shutil.copy2(path,snap/'diffusion-dashboard'/path.name)
    shutil.copy2(ROOT/'train.py',snap/'train.py')
    (snap/'configs').mkdir()
    data=out/'data/processed/160m-k4p-c48_square_safe'
    base=ROOT/'data/processed/160m-k4p-c48'
    linked=[]
    for split in ('train','val','test'):
        (data/split).mkdir(parents=True)
        for source in (base/split).iterdir():
            if not source.is_file() or source.name.startswith('alm_'): continue
            target=data/split/source.name
            # Arrays are immutable inputs; regenerated ALM arrays are independent files.
            if source.suffix=='.npy': os.link(source,target)
            else: shutil.copy2(source,target)
            linked.append(dict(source=str(source),target=str(target),size=source.stat().st_size))
    for source in base.iterdir():
        if source.is_file() and source.name!='alm_constraints_report.json': shutil.copy2(source,data/source.name)
    configs={}
    for tag,name in [('D','paper_modified_d.yaml'),('S','paper_modified_s.yaml'),('A','paper_modified.yaml')]:
        cfg=load_config(str(ROOT/'configs'/name))
        assert not cfg['model']['path_features']['enabled']
        assert not cfg['model']['safety_query']['center_encoding']
        cfg['alm']['final_project']=False
        cfg['data']['processed_root']=str(data)
        cfg['train'].update(epochs=200,max_hours=None,ckpt_dir=str(out/tag/'ckpt'))
        path=snap/'configs'/f'{tag}.yaml'
        path.write_text(yaml.safe_dump(cfg,sort_keys=False,allow_unicode=True),encoding='utf-8')
        configs[tag]=str(path)
    # Keep the original 72 task conditions fixed, without reusing any outcomes.
    old=ROOT/'outputs/chapter4/random_endpoints/results_seed20260927.json'
    if old.exists():
        tasks=json.loads(old.read_text(encoding='utf-8'))
        for row in tasks['records']:
            row.pop('result',None); row.pop('error',None); row.pop('http_status',None)
        tasks.pop('summary',None)
        write(out/'inputs/random_endpoints_tasks.json',tasks)
    hashes={str(p.relative_to(snap)):hashlib.sha256(p.read_bytes()).hexdigest() for p in snap.rglob('*') if p.is_file()}
    manifest=dict(python=sys.executable,source_sha256=hashes,configs=configs,
        original_dataset=str(base),processed_dataset=str(data),immutable_input_links=linked,
        architecture='no Hpath, no additive center encoding; centers sampled along route',
        final_project=False,excluded_baseline='projection',train_epochs=200,
        checkpoint_selection='best_task using validation only',seeds=[0,1,2],
        methods=['none','penalty','primal_dual','alm_sdf','alm'],
        requires_retraining=['D','S','A'],
        reason='old offline ALM constraints overlap occupied pixel squares; labels rebuilt with corrected geometry',
        started_at=datetime.datetime.now().isoformat())
    write(out/'manifest.json',manifest)
    return manifest

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--out',required=True)
    ap.add_argument('--resume',action='store_true')
    ap.add_argument('--prepare-only',action='store_true')
    args=ap.parse_args()
    out=Path(args.out).resolve()
    manifest=json.loads((out/'manifest.json').read_text(encoding='utf-8')) if args.resume else prepare(out)
    if args.prepare_only: print(out,flush=True); return
    snap=out/'source_snapshot'
    ledger=json.loads((out/'completed_stages.json').read_text(encoding='utf-8')) if (out/'completed_stages.json').exists() else {}
    env=dict(os.environ,OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',PYTHONUNBUFFERED='1',PYTHONUTF8='1')
    def run(stage,script,argv):
        if stage in ledger: return
        log=out/'logs'/f'{stage}.log'; log.parent.mkdir(parents=True,exist_ok=True)
        command=[sys.executable,'-u',str(snap/script),*[str(a) for a in argv]]
        write(out/'status.json',dict(stage=stage,updated_at=datetime.datetime.now().isoformat(),log=str(log),command=command))
        print('RUN',stage,flush=True)
        with log.open('a',encoding='utf-8') as handle:
            subprocess.run(command,cwd=snap,env=env,stdout=handle,stderr=subprocess.STDOUT,check=True)
        ledger[stage]=dict(completed_at=datetime.datetime.now().isoformat(),log=str(log))
        write(out/'completed_stages.json',ledger)
    try:
        data=manifest['processed_dataset']; configs=manifest['configs']
        run('rebuild_constraints','scripts/data/carla_full/04_build_alm_constraints.py',
            ['--processed',data,'--config',configs['A'],'--workers','6'])
        run('audit_constraints','scripts/audit_paper_constraints.py',
            ['--processed',data,'--out',out/'offline_constraint_audit.json','--workers','6'])
        for tag in ('D','S','A'):
            argv=['--config',configs[tag],'--ckpt-dir',out/tag/'ckpt','--out-dir',out/tag,
                  '--epochs','200','--max-restarts','3','--restart-sleep','30']
            if (out/tag/'ckpt/latest.pt').exists(): argv+=['--resume',out/tag/'ckpt/latest.pt']
            run('train_'+tag,'scripts/night_train.py',argv)
            summary=json.loads((out/tag/'training_summary.json').read_text(encoding='utf-8'))
            if summary['epochs_done']!=200 or not (out/tag/'ckpt/best_task.pt').exists():
                raise RuntimeError(f'{tag} did not complete 200 epochs with a validation-selected checkpoint')
        winners={}
        for mode in ('penalty','primal_dual','alm_sdf'):
            path=out/'sweeps'/f'{mode}.json'
            run('sweep_'+mode,'scripts/ch4_sweep_val.py',
                ['--config',configs['D'],'--ckpt',out/'D/ckpt/best_task.pt','--guidance',mode,
                 '--out',path,'--work-dir',out/'sweeps'/mode,'--val-samples','256','--fix-inner-steps','10'])
            winner=json.loads(path.read_text(encoding='utf-8'))
            if winner['winner_aggregate'] is None: raise RuntimeError('All sweep candidates failed')
            winners[mode]=winner['winner']
        def evaluate(group,name,tag,mode,feedback='off',safe='on',steps=16,stride=1):
            for seed in manifest['seeds']:
                stem=out/group/f'{name}_seed{seed}'
                stem.parent.mkdir(parents=True,exist_ok=True)
                argv=['--config',configs[tag],'--ckpt',out/tag/'ckpt/best_task.pt',
                    '--method',name,'--guidance',mode,'--feedback',feedback,'--safe-to-ddim',safe,
                    '--final-project','off','--steps',steps,'--split','test','--samples','0',
                    '--chunk','32','--seed',seed,'--out',str(stem)+'.json',
                    '--records',str(stem)+'.jsonl','--md',str(stem)+'.md']
                if stride!=1:
                    cfg=load_config(configs[tag]); cfg['corridor']['region_stride']=stride
                    import yaml
                    variant=out/'derived_configs'/f'{tag}_stride{stride}.yaml'; variant.parent.mkdir(exist_ok=True)
                    variant.write_text(yaml.safe_dump(cfg,sort_keys=False),encoding='utf-8')
                    argv[1]=variant
                for key,value in winners.get(mode,{}).items(): argv+=['--sdf-set',f'{key}={value}']
                run(f'{group}_{name}_{seed}','scripts/eval_chapter4.py',argv)
                payload=json.loads(Path(str(stem)+'.json').read_text(encoding='utf-8'))
                if payload['final_project'] or payload['aggregate']['n']!=420:
                    raise RuntimeError('Evaluation protocol mismatch')
        for mode in ('none','penalty','primal_dual','alm_sdf'):
            evaluate('main',mode,'D',mode)
        evaluate('main','TrajSafe','A','alm','on')
        evaluate('progressive','S_raw','S','none')
        evaluate('progressive','A_feedback_only','A','alm','on','off')
        # D raw and A full are reused from the main table: identical model, flags and noise.
        write(out/'progressive/reused_rows.json',dict(D_raw='../main/none_seed*.json',A_full='../main/TrajSafe_seed*.json'))
        for steps,stride in ((4,1),(16,8),(4,8)):
            evaluate('efficiency_quality',f'steps{steps}_stride{stride}','A','alm','on',steps=steps,stride=stride)
        run('timing','scripts/paper_modified_timing.py',['--run',out])
        run('geometry','scripts/paper_modified_geometry.py',['--run',out])
        run('custom_endpoints','scripts/paper_modified_endpoints.py',['--run',out])
        groups={}
        import numpy as np
        for group in ('main','progressive','efficiency_quality'):
            for path in sorted((out/group).glob('*_seed0.json')):
                name=path.stem[:-6]; records=[]
                for seed in manifest['seeds']:
                    records += [json.loads(line) for line in (out/group/f'{name}_seed{seed}.jsonl').read_text(encoding='utf-8').splitlines() if line]
                if len(records)!=1260: raise RuntimeError('Expected 1260 paired trajectories')
                groups[f'{group}/{name}']=dict(n=len(records),collisions=sum(r['collision'] for r in records),
                    rmse_m=float(np.mean([r['rmse_m'] for r in records])),smoothness_m=float(np.mean([r['smoothness_m'] for r in records])))
        write(out/'results.json',groups)
        write(out/'status.json',dict(stage='complete',updated_at=datetime.datetime.now().isoformat()))
    except Exception:
        write(out/'status.json',dict(stage='failed',updated_at=datetime.datetime.now().isoformat(),traceback=traceback.format_exc()))
        raise

if __name__=='__main__': main()
