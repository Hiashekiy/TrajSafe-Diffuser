"""Optional raw-prediction and projection diagnostics for architecture evaluation."""
import numpy as np
import torch


def collect(model, out, occ, cond, batch, sdf, scene_to_meter, evaluate_controls):
    # The last network prediction before ALM, on the same guided sampling chain.
    trace = out.get('trace') or []
    if not trace:
        raise RuntimeError('Raw diagnostics require return_trace=True')
    raw_q = trace[-1]['q0_raw'].to(occ.device)
    raw = evaluate_controls(model.bspline, raw_q, occ, cond,
                            batch['pos'].to(occ.device), sdf, None, batch, scene_to_meter)
    curve_change = model.bspline.decode_controls(out['control'] - raw_q).norm(dim=-1).amax(dim=-1) * scene_to_meter
    projection = out.get('final_projection') or {}
    applied = projection.get('hard_project_applied', torch.zeros(len(raw_q), dtype=torch.bool))
    corrections = [frame['alm_correction'] for frame in trace if frame.get('alm_correction') is not None]
    alm_change = torch.stack(corrections).amax(dim=0) * scene_to_meter if corrections else torch.zeros(len(raw_q))
    records = []
    for i in range(len(raw_q)):
        records.append({
            'raw_collision': bool(raw['collision'][i]),
            'raw_sdf_feasible': bool(raw['sdf_feasible'][i]),
            'raw_sdf_violation_m': float(raw['sdf_violation_m'][i]),
            'raw_rmse_m': float(raw['rmse_m'][i]),
            'raw_smoothness_m': float(raw['smoothness_m'][i]),
            'raw_to_final_max_curve_change_m': float(curve_change[i]),
            'max_step_alm_curve_change_m': float(alm_change[i]),
            'final_projection_applied': bool(applied[i]),
        })
    return records


def summarize(records):
    return {key: float(np.mean([row[key] for row in records]))
            for key in records[0]} if records else {}
