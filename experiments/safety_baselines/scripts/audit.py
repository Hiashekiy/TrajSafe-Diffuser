"""Read-only asset/Git audit. Writes only beneath this experiment."""
from pathlib import Path
import hashlib,json,subprocess,platform,sys
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parents[1]
def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()
def main():
    assets={}
    for rel in ['outputs/models/oneshot_k4p_c48_d/ckpt/best_task.pt','outputs/models/v4_map16_c48_l1/ckpt/best_task.pt','configs/config_160k4p_c48_d_std.yaml','configs/config_160k4p_c48_v4_map16_l1.yaml']:
        p=ROOT/rel;assets[rel]={'exists':p.exists(),'sha256':sha(p) if p.exists() else None}
    papers=ROOT/'docs/TrajSafe_Safety_Baselines_Codex_Kit/experiments/safety_baselines/papers'
    for p in papers.glob('*.pdf'):assets[str(p.relative_to(ROOT))]={'sha256':sha(p),'bytes':p.stat().st_size}
    git={cmd:subprocess.run(['git',*cmd.split()],cwd=ROOT,capture_output=True,text=True,encoding='utf8').stdout for cmd in ['status --short','branch --show-current','rev-parse HEAD','log -5 --oneline']}
    data={'python':sys.version,'executable':sys.executable,'platform':platform.platform(),'git':git,'assets':assets,'ours_selected_by_user':'v4_map16_l1'}
    (HERE/'docs/audit_manifest.json').write_text(json.dumps(data,ensure_ascii=False,indent=2),encoding='utf8')
    print('audit_manifest.json written')
if __name__=='__main__':main()
