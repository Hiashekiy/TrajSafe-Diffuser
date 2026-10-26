"""Repeat fixed custom-endpoint tasks directly through the new model's engine."""
import argparse,copy,json,sys,time,traceback
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT));sys.path.insert(0,str(ROOT/'scripts'));sys.path.insert(0,str(ROOT/'diffusion-dashboard'))

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',required=True);args=ap.parse_args()
    target=Path(args.run);m=json.loads((target/'manifest.json').read_text(encoding='utf-8'))
    import engine_carla as e
    from eval_random_endpoints import compact_result,summarize
    e.MODELS['paper_modified']={'config':m['configs']['A'],'ckpt_dir':str(target/'A/ckpt'),'label':'Paper modified'}
    model_id='paper_modified:best_task';e.CHECKPOINTS[model_id]=str(target/'A/ckpt/best_task.pt')
    engine=e.Engine('cuda',processed_root=m['processed_dataset'],config_path=m['configs']['A'],dataset='160k4p_c48')
    occ=np.load(Path(m['processed_dataset'])/'test/occupancy.npy',mmap_mode='r')
    output=target/'custom_endpoints';output.mkdir(exist_ok=True)
    def generate(index,condition,seed,key):
        begin=time.perf_counter()
        payload=engine.generate(key,'test',index,np.asarray(occ[index]).squeeze(),condition,seed,
            model_id=model_id,alm_enabled=True,steps=16,dataset='160k4p_c48',final_project=False)
        payload['elapsed_ms']=1000*(time.perf_counter()-begin)
        if payload['alm'].get('final_project'):raise RuntimeError('Terminal projection enabled')
        return payload
    condition=[[0.8948432803153992,-0.8210313320159912],[0.9211324453353882,0.8169868588447571]]
    case=generate(293,condition,43,'paper_case293')
    (output/'case293.json').write_text(json.dumps(case,ensure_ascii=False),encoding='utf-8')
    import plot_dashboard_case293 as plot
    plot.load_payload=lambda:case;plot.OCCUPANCY=Path(m['processed_dataset'])/'test/occupancy.npy'
    plot.OUTPUT=output/'case293.png';plot.main()
    tasks=json.loads((target/'inputs/random_endpoints_tasks.json').read_text(encoding='utf-8'))
    if len(tasks['records'])!=72:raise RuntimeError('Expected original 72 fixed endpoint tasks')
    for i,row in enumerate(tasks['records']):
        try:
            payload=generate(row['map_index'],row['condition'],43,f'paper_random_{i}')
            row['result']=compact_result(payload);row['http_status']=200 # Legacy plot schema: successful generation.
        except Exception:
            row['error']=traceback.format_exc();row['http_status']=500
        print('endpoint',i+1,'/72',row['http_status'],flush=True)
        (output/'random_endpoints.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    tasks['summary']=summarize(tasks['records'])
    tasks['protocol'].update(model=model_id,model_id=model_id,final_project=False,transport='direct Python engine')
    (output/'random_endpoints.json').write_text(json.dumps(tasks,ensure_ascii=False,indent=2),encoding='utf-8')
    import subprocess
    for pair in ('0,1','2,3','4,5'):
        subprocess.run([sys.executable,str(ROOT/'scripts/plot_random_endpoints.py'),'--results',str(output/'random_endpoints.json'),
            '--pair-indices',pair,'--occupancy',str(Path(m['processed_dataset'])/'test/occupancy.npy'),
            '--out',str(output/f'random_endpoints_{pair.replace(",","_")}.png')],check=True,cwd=ROOT)

if __name__=='__main__':main()
