"""Regenerate the aligned configs from the frozen schedule (no hand-typed constants).

Derived quantities, all from src.diffusion.schedule:
  pdm_ald_eps : official ALD uses delta_t = eps*(sigma_t/sigma_min)^2 and the paper's
                gamma is delta_t/2, so gamma_t/sigma_t^2 = eps/(2 sigma_min^2).
                eps is fixed by the required per-step mixing ratio 1/6.
  dual_eta    : paper Appendix C "large" eta_lambda = 1e-3, and with g' = s*g the
                equivalent scaled parameter is eta' = eta/s^2 (s = dual_constraint_scale).
  rho         : same rule for ALM, rho' = rho/s^2 (paper "large" rho = 2.5e-4).
  cd_levels   : every trained level is visited (cd_beta="level" is only exact then).
"""
import json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT))
import numpy as np
from src.diffusion.schedule import NoiseSchedule

SCALE=128.0
PAPER_ETA=1e-3
PAPER_RHO=2.5e-4 if False else 2.5e-4
MIXING=1.0/6.0

def derived(timesteps):
    s=NoiseSchedule(timesteps,'squaredcos_cap_v2');ab=s.alphas_cumprod.numpy();sig=np.sqrt((1-ab)/ab)
    smin2=float(sig[0]**2)
    return {'timesteps':timesteps,'sigma_min_scene':float(sig[0]),'sigma_min_m':float(sig[0]*80),
        'sigma_max_scene':float(sig[-1]),'smin2':smin2,
        'pdm_ald_eps':round(MIXING*2.0*smin2,10),
        'dual_eta':PAPER_ETA/SCALE**2,'rho':PAPER_RHO/SCALE**2,
        'langevin_floor_rmse_m':float(np.sqrt(2)*sig[0]*80)}

BASE=ROOT/'experiments/safety_baselines/configs'

def build(timesteps,checkpoint,training_run,segment_for,out_name,also_note):
    d=derived(timesteps)
    cfg=json.loads((BASE/'seven_native_points_template.json').read_text(encoding='utf8'))
    cfg['checkpoint']=checkpoint
    cfg['pdm_ald_eps']=d['pdm_ald_eps']
    cfg['dual_eta']=round(d['dual_eta'],12)
    cfg['rho']=round(d['rho'],14)
    cfg['cd_levels']=timesteps
    cfg['cd_beta']='level'
    cfg['pdm_parameterization']='ve'
    cfg['segment_constraints_for']=segment_for
    cfg['training_run']=training_run
    cfg['schedule_facts']={k:(round(v,10) if isinstance(v,float) else v) for k,v in d.items()}
    cfg['adaptations'].update({
      'pdm_parameterization':'ve (paper state y). 2026-11-14: the VP-state variant was removed because it projected the sqrt(alphabar_t)-scaled state onto the *unscaled* set C, and did not carry y across levels; that is neither the paper VE algorithm nor an exact reparameterisation (docs/DSH_FIX_REPORT_20261114.md §4d).',
      'pdm_ald_eps':'derived: official ALD delta_t = eps*(sigma/sigma_min)^2 with the paper form gamma=delta/2 gives gamma/sigma^2 = eps/(2 sigma_min^2); eps fixed so that ratio = 1/6. Fixes the earlier factor-of-2 (gamma=delta_t) error.',
      'dual_eta':'paper Appendix C "large" eta_lambda=1e-3, divided by dual_constraint_scale^2=16384 because g was scaled by 128 (equivalence: lambda\'=lambda/s, eta\'=eta/s^2; test_dual_units_invariance).',
      'rho':'paper Appendix C "large" rho=2.5e-4 divided by dual_constraint_scale^2=16384 (rho\'=rho/s^2).',
      'cd_levels':'every trained level is visited: eq.(6) uses that level\'s own beta_t, which is only correct when the ladder is not subsampled.',
      'cd_beta':'level',
      'pdm_inner':'20 instead of M=100 (PDM App.F explicitly permits reducing M for expensive CPU projections).',
      'segment_constraints_for':also_note})
    (BASE/out_name).write_text(json.dumps(cfg,indent=2,ensure_ascii=False),encoding='utf8')
    print(out_name,'T=%d'%timesteps,'eps=%.4g'%d['pdm_ald_eps'],'eta=%.4g'%d['dual_eta'],'rho=%.4g'%d['rho'],
        'sigma_min=%.4f m'%d['sigma_min_m'],'floor=%.2f m'%d['langevin_floor_rmse_m'],'segments for',segment_for)

NOTE_SEG='paper-native: PDM uses App.C.3 whole-segment constraints; the other six keep their own paper mechanism on the discrete waypoint state.'
NOTE_PT='diagnostic variant: all seven use per-vertex free-space rows, so it is NOT a paper-native PDM run (see Problem B in docs/DSH_FIX_REPORT_20261114.md §4e).'

build(256,'experiments/safety_baselines/results/map_only_points_t256_e100/ckpt/best_task.pt',
      'experiments/safety_baselines/results/map_only_points_t256_e100',['PDM'],
      'seven_native_points_paper_t256.json',NOTE_SEG)
build(256,'experiments/safety_baselines/results/map_only_points_t256_e100/ckpt/best_task.pt',
      'experiments/safety_baselines/results/map_only_points_t256_e100',[],
      'seven_native_points_vertexall_t256.json',NOTE_PT)
build(16,'experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521/ckpt/best_task.pt',
      'experiments/safety_baselines/results/map_only_points_l1_final8_e100_20261114_000521',[],
      'seven_native_points_vertexall_t16.json',NOTE_PT)
