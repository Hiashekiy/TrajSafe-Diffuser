"""Train and evaluate only the modified model, using a frozen source snapshot."""
from pathlib import Path
import datetime
import hashlib
import json
import shutil
import subprocess
import sys
import traceback
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.utils.config import load_config

OUT = ROOT / 'outputs/ablation_hpath/without_hpath'
SNAP = OUT / 'source_snapshot'
CFG = SNAP / 'configs/modified.yaml'
STATUS = OUT / 'status.json'


def status(stage, **extra):
    payload = {'stage': stage, 'updated_at': datetime.datetime.now().isoformat(), **extra}
    tmp = STATUS.with_suffix('.tmp')
    tmp.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding='utf-8')
    tmp.replace(STATUS)
    print(json.dumps(payload, ensure_ascii=False), flush=True)


def run(argv, log):
    print('RUN', subprocess.list2cmdline([str(x) for x in argv]), flush=True)
    with log.open('a', encoding='utf-8') as handle:
        subprocess.run([str(x) for x in argv], cwd=SNAP, stdout=handle,
                       stderr=subprocess.STDOUT, check=True)


def freeze():
    OUT.mkdir(parents=True, exist_ok=True)
    if SNAP.exists():
        raise RuntimeError('Source snapshot already exists; avoid overwriting experiment provenance')
    SNAP.mkdir()
    shutil.copytree(ROOT / 'src', SNAP / 'src', ignore=shutil.ignore_patterns('__pycache__', '*.pyc'))
    shutil.copy2(ROOT / 'train.py', SNAP / 'train.py')
    (SNAP / 'scripts').mkdir()
    for name in ('eval_chapter4.py', 'hpath_diagnostics.py'):
        shutil.copy2(ROOT / 'scripts' / name, SNAP / 'scripts' / name)
    (SNAP / 'configs').mkdir()
    cfg = load_config(str(ROOT / 'configs/ablation_hpath_without.yaml'))
    assert cfg['model']['path_features']['enabled'] is False
    assert cfg['model']['safety_query']['center_encoding'] is False
    cfg['data']['processed_root'] = str(ROOT / 'data/processed/160m-k4p-c48')
    cfg['train']['ckpt_dir'] = str(OUT / 'ckpt')
    cfg['train']['epochs'] = 200
    cfg['train']['max_hours'] = None
    import yaml
    CFG.write_text(yaml.safe_dump(cfg, allow_unicode=True, sort_keys=False), encoding='utf-8')
    hashes = {str(p.relative_to(SNAP)): hashlib.sha256(p.read_bytes()).hexdigest()
              for p in SNAP.rglob('*') if p.is_file()}
    manifest = {'python': sys.executable, 'source_sha256': hashes,
                'training_seed': cfg['env']['seed'], 'generation_seeds': [0, 1, 2],
                'train_epochs': 200, 'checkpoint_selection': 'best_task on validation',
                'dataset': cfg['data']['processed_root'],
                'architecture': 'no H_path; no additive center encoding; direct control-to-safety attention',
                'scope': 'one modified model only; two architecture changes together; no isolated causal attribution'}
    (OUT / 'manifest.json').write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding='utf-8')


def report():
    records = []
    for seed in (0, 1, 2):
        path = OUT / 'evaluation' / ('seed%d.jsonl' % seed)
        rows = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line]
        if len(rows) != 420:
            raise RuntimeError('Expected 420 records per seed, got %d' % len(rows))
        records.extend(rows)
    def mean(k):
        return float(np.mean([r[k] for r in records]))
    metrics = {'n': len(records), 'collisions': sum(r['collision'] for r in records),
               'collision_rate': mean('collision'), 'sdf_feasible_rate': mean('feasible'),
               'mean_violation_m': mean('nonnegative_violation'),
               'p95_violation_m': float(np.percentile([r['nonnegative_violation'] for r in records], 95)),
               'rmse_m': mean('rmse_m'), 'smoothness_m': mean('smoothness_m'),
               'raw_collision_rate': mean('raw_collision'),
               'raw_sdf_feasible_rate': mean('raw_sdf_feasible'),
               'raw_mean_violation_m': mean('raw_sdf_violation_m'),
               'raw_rmse_m': mean('raw_rmse_m'), 'raw_smoothness_m': mean('raw_smoothness_m'),
               'final_projection_applied_rate': mean('final_projection_applied'),
               'max_step_alm_curve_change_m_mean': mean('max_step_alm_curve_change_m'),
               'raw_to_final_max_curve_change_m_mean': mean('raw_to_final_max_curve_change_m')}
    (OUT / 'results.json').write_text(json.dumps(metrics, indent=2), encoding='utf-8')
    summary = json.loads((OUT / 'training_summary.json').read_text(encoding='utf-8'))
    lines = ['# Modified TrajSafe model evaluation', '',
             'Only the modified model was trained. H_path and additive center encoding are both removed.',
             'Dataset: carla_processed_160k4p_c48; 420 test samples x generation seeds 0/1/2 = 1260 trajectories.',
             'DDIM: 16 steps; warmup: 3 steps; feedback, ALM and final projection enabled.',
             'Raw prediction means the last network prediction before ALM on the SAME guided chain.',
             'This run tests the combined modified architecture and does not isolate either change.', '',
             'Training epochs: %d; parameters: %.3f M; selected epoch (zero-based): %s.' %
             (summary['epochs_done'], summary['params_m'], summary['best_task_epoch']), '',
             '| Metric | Value |', '|---|---:|']
    lines += ['| %s | %.6f |' % (k, v) for k, v in metrics.items()]
    lines += ['', 'Rates are fractions. Timing is stored per seed in evaluation/*.json and is batch throughput, not request latency.']
    (OUT / 'REPORT.md').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    return metrics


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    try:
        freeze()
        status('training', epochs=200, log=str(OUT / 'training.log'))
        run([sys.executable, '-u', SNAP / 'train.py', '--config', CFG,
             '--device', 'cuda', '--out-dir', OUT], OUT / 'training.log')
        summary = json.loads((OUT / 'training_summary.json').read_text(encoding='utf-8'))
        if summary['epochs_done'] != 200:
            raise RuntimeError('Training did not finish all 200 epochs')
        ckpt = OUT / 'ckpt/best_task.pt'
        if not ckpt.is_file():
            raise RuntimeError('Validation-selected checkpoint missing')
        (OUT / 'evaluation').mkdir(exist_ok=True)
        for seed in (0, 1, 2):
            status('evaluating', seed=seed, checkpoint=str(ckpt))
            run([sys.executable, '-u', SNAP / 'scripts/eval_chapter4.py', '--config', CFG,
                 '--ckpt', ckpt, '--method', 'Modified-NoHpath-NoCenterEncoding',
                 '--guidance', 'alm', '--feedback', 'on', '--final-project', 'on',
                 '--split', 'test', '--samples', '0', '--chunk', '32', '--steps', '16',
                 '--seed', str(seed), '--device', 'cuda', '--raw-diagnostics',
                 '--out', OUT / 'evaluation' / ('seed%d.json' % seed)], OUT / 'evaluation.log')
        metrics = report()
        status('complete', report=str(OUT / 'REPORT.md'), metrics=metrics)
    except Exception as exc:
        status('failed', error=str(exc), traceback=traceback.format_exc())
        raise


if __name__ == '__main__':
    main()
