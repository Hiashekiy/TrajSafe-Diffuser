import time,hashlib
import numpy as np
import torch
from experiments.safety_baselines.adapters.paper_task import PaperTask
from experiments.safety_baselines.algorithms.solvers import Budget
from experiments.safety_baselines.algorithms.safediffuser.runner import run as safe_run
from experiments.safety_baselines.algorithms.constrained_diffusers.runner import run as cd_run
from experiments.safety_baselines.algorithms.pdm.runner import run as pdm_run
METHODS=['RoS-diffuser','ReS-diffuser','TVS-diffuser','Projected','Primal-Dual','ALM','PDM']
SOURCES={m:('SafeDiffuser ICLR2025' if 'diffuser' in m else 'PDM NeurIPS2024' if m=='PDM' else 'Constrained Diffusers NeurIPS2025') for m in METHODS}

def derived_seed(scene_id,seed,method):return int.from_bytes(hashlib.sha256(f'{scene_id}|{seed}|{method}'.encode()).digest()[:8],'little')

def initial_seed(scene_id,seed):return int.from_bytes(hashlib.sha256(f'{scene_id}|{seed}|initial'.encode()).digest()[:8],'little')

def initial_noise(scene_id,seed,num_points=128):
    """The single trajectory noise vector every method starts from (paired protocol)."""
    return np.random.default_rng(initial_seed(scene_id,seed)).normal(size=(num_points,2))

def attach_nfe_counter(task):
    """Count REAL network forward passes on `task.clean`.

    The NFE of these loops is not `cd_levels`: Projected/PD/ALM run
    (levels + cd_extra_steps) updates and PDM runs pdm_levels * pdm_inner score
    evaluations. (2026-11-14 review item 4.)
    """
    counter=[0];inner=task.clean
    def counted(x,t,ve=False):
        counter[0]+=1;return inner(x,t,ve)
    task.clean=counted
    return counter

def run_method(method,model,schedule,occ,cond,scene_id,seed,cfg,init_noise=None,task_extra=None):
    started=time.perf_counter();trace=[];trajectory=None;reason=None;status='ok'
    try:
        budget=Budget(cfg.get('timeouts',{}).get(method,cfg['timeout_seconds']))
        Task=PaperTask;extra={}
        if cfg.get('state_adapter')=='native_points':
            from experiments.safety_baselines.adapters.native_points import NativePointTask
            from experiments.safety_baselines.adapters.segment_points import SegmentPointTask
            Task=SegmentPointTask if method in cfg.get('segment_constraints_for',[]) else NativePointTask
            if Task is SegmentPointTask:
                extra={'margin':cfg.get('segment_circle_margin_scene',0.)}
        elif cfg.get('state_adapter')=='curve_control':
            from experiments.safety_baselines.adapters.curve_control import CurveControlTask
            Task=CurveControlTask
        elif cfg.get('state_adapter')=='sc48_control':
            from experiments.safety_baselines.adapters.sc48_control import SC48ControlTask
            Task=SC48ControlTask;extra=dict(task_extra or {})
        elif cfg.get('state_adapter')=='physical_trajectory':
            from experiments.safety_baselines.adapters.physical_trajectory import PhysicalTrajectoryTask
            Task=PhysicalTrajectoryTask
        elif cfg.get('state_adapter')=='trajectory':
            from experiments.safety_baselines.adapters.trajectory_state import TrajectoryStateTask
            Task=TrajectoryStateTask
        if cfg.get('state_adapter')=='sc48_control':extra=dict(task_extra or {})
        task=Task(model,schedule,occ,cond,cfg['constraint_points'],cfg['radius_scene'],cfg.get('field_kind','bilinear'),**extra)
        if hasattr(task,'projection_maxiter'):
            task.projection_maxiter=int(cfg.get('segment_projection_maxiter',8))
            task.projection_maxiter_final=int(cfg.get('segment_projection_maxiter_final',40))
            task.projection_tol=float(cfg.get('segment_projection_tol',1e-6))
        if cfg.get('state_adapter') in ('curve_control','sc48_control'):
            task.projection_maxiter=cfg['projection_maxiter'];task.projection_tol=cfg['projection_tol_scene']
        # Count REAL network forward passes (see attach_nfe_counter).
        nfe=attach_nfe_counter(task)
        if init_noise is not None:
            task.initial_override=np.asarray(init_noise,dtype=float)[1:-1].reshape(-1)
            task.particle_seed=initial_seed(scene_id,seed)   # method independent, pairs PD/ALM
        rng=np.random.default_rng(derived_seed(scene_id,seed,method));fn=safe_run if method.endswith('diffuser') else pdm_run if method=='PDM' else cd_run
        value=fn(task,method,rng,cfg,budget,trace);budget.check();trajectory=task.unpack(value)
        if not np.isfinite(trajectory).all():raise RuntimeError('nonfinite_trajectory')
    except (RuntimeError,ValueError,np.linalg.LinAlgError) as exc:status='failed';reason=str(exc);trajectory=None
    torch.cuda.synchronize()
    return {'trajectory':None if trajectory is None else trajectory.tolist(),
        'trajectory_before':task.unpack(task.last_nominal).tolist() if 'task' in locals() and hasattr(task,'last_nominal') and np.isfinite(task.last_nominal).all() else None,
        'status':status,'failure_reason':reason,'wall_time':time.perf_counter()-started,'per_step_trace':trace,
        'method':method,'source_id':SOURCES[method],'fidelity':'task-adapted',
        'seed_derived':derived_seed(scene_id,seed,method),'initial_seed':initial_seed(scene_id,seed),
        'paired_initial_noise':init_noise is not None,
        'pdm_projection_stats':getattr(task,'pdm_stats',None),
        'segment_audit':getattr(task,'segment_audit',None),
        'internal_state_trajectory':task.points(value).tolist() if status=='ok' and hasattr(task,'points') else None,
        'internal_max_violation_scene':float(task.constraints(value)[0].max()) if status=='ok' else None,
        'model_forward_calls':nfe[0]}
