"""Production sampler, batch=1, final projected Q and dense verification."""
import hashlib
import sys
from pathlib import Path as FilePath
import numpy as np
from time import perf_counter

ROOT = FilePath(__file__).resolve().parents[2]


class PlannerAdapter:
    def __init__(self, cfg):
        import torch
        sys.path.insert(0, str(ROOT/'diffusion-dashboard'))
        from engine_carla import Engine, CHECKPOINTS, model_config
        from src.utils.config import num_controls
        self.cfg = cfg
        self.viz = bool(cfg.get('visualize'))
        self.model_id = cfg['model_id']
        self.engine = Engine(cfg['device'], config_path=str(ROOT/cfg['config']),
                             dataset=cfg.get('dataset', '160m-k4p-c48'))
        self.model, self.epoch = self.engine.get_model(self.model_id)
        self.model_cfg = model_config(self.model_id)
        from src.geometry.skeleton_paths import CandidateConfig
        self.online_cand_cfg = CandidateConfig.from_dict(dict(self.engine.cfg['topology'],
            raw_k=cfg.get('raw_topology_candidates', 16)), strict=False)
        checkpoint = FilePath(CHECKPOINTS[self.model_id])
        ck = torch.load(checkpoint, map_location='cpu', weights_only=False)
        meta = ck['config']
        if num_controls(meta) != 48 or float(meta['data']['scene_to_meter']) != 80:
            raise ValueError('CHECKPOINT_COORDINATE_CONTRACT_MISMATCH')
        stored, actual = ck['model_state'], self.model.state_dict()
        if set(stored) != set(actual) or any(stored[k].shape != actual[k].shape for k in actual):
            raise ValueError('CHECKPOINT_REQUIRES_PARAMETER_ADAPTATION')
        self.manifest = dict(checkpoint=str(checkpoint), checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
                             epoch=self.epoch, controls=48, scene_to_meter=80, map_size_m=160,
                             training_obstacle_transform='erosion k=4, NOT inflation',
                             online_obstacle_transform=f"actual road; actor boxes inflated longitudinal {cfg.get('inflation_cells',4)} cells, lateral {cfg.get('inflation_lateral_cells',cfg.get('inflation_cells',4))} cells")
        self.manifest['online_adaptation'] = dict(query_branch_subdivision=True,
            raw_topology_candidates=self.online_cand_cfg.raw_k,
            geometric_output='learned Q in the frozen corridor; no online-only constraints')
        self.manifest['resolved_model_config'] = self.model_cfg

    def __call__(self, request):
        import torch
        from src.diffusion.sampler import sample, effective_warmup_steps
        from src.geometry.skeleton_paths import generate_candidates
        from .path_profile import Path
        from .skeleton_query import query_graph
        engine = self.engine
        occ, condition = request['occupancy'], request['condition']
        t0 = perf_counter()
        graph = engine.get_graph(engine.occupancy_key(occ), occ)
        graph = query_graph(graph, *condition)
        cands = generate_candidates(graph, condition[0], condition[1], self.online_cand_cfg)
        if cands.num_valid == 0:
            raise ValueError('NO_GEOMETRIC_CANDIDATE')
        packed = engine._pack_candidates(cands, condition)
        viz_candidates = None
        if self.viz:
            viz_candidates = dict(xy=packed['xy'][0].float().cpu().numpy(),
                                  mask=packed['mask'][0].cpu().numpy().astype(bool),
                                  geometry=packed['geometry'][0].float().cpu().numpy(),
                                  lengths=packed['geometry_lengths'][0].cpu().numpy())
        topology_ms = 1000*(perf_counter()-t0)
        device = self.cfg['device']
        alm = dict(self.model_cfg['alm'])
        corridor = dict(self.model_cfg['corridor'])
        if self.cfg.get('corridor_margin_m') is not None:
            # 安全裕度加在【走廊构建】上（局部收紧），而不是膨胀障碍（全局侵蚀自由空间）
            region = dict(corridor.get('region') or {})
            region['safety_margin'] = float(self.cfg['corridor_margin_m'])/80.0
            corridor['region'] = region
        if self.cfg.get('corridor_axis_max_scene') is not None:
            region = dict(corridor.get('region') or {})
            region['ellipse_axis_max'] = float(self.cfg['corridor_axis_max_scene'])
            corridor['region'] = region
        if self.cfg.get('region_stride') is not None:
            # Online-only region sharing; the offline evaluation keeps stride 1.
            corridor['region_stride'] = int(self.cfg['region_stride'])
        alm['warmup_reverse_steps'] = effective_warmup_steps(alm.get('warmup_reverse_steps', 3), self.cfg['steps'])
        cond = torch.as_tensor(condition[None], dtype=torch.float32, device=device)
        occ_tensor = torch.as_tensor(occ[None, None], dtype=torch.float32, device=device)
        with torch.inference_mode():
            out = sample(self.model, engine.schedule, cond, occ_tensor, packed['xy'], packed['mask'],
                         packed['geometry'], packed['geometry_lengths'], device=device, steps=self.cfg['steps'],
                         seed=request['seed'], return_trace=self.viz, alm_config=alm,
                         corridor_config=corridor)
        controls = out['control'][0].detach().cpu().numpy()
        scene = out['p'][0].detach().cpu().numpy()
        out['timing']['topology_s'] = topology_ms/1000
        if controls.shape != (48, 2) or not np.isfinite(controls).all():
            raise ValueError('INVALID_CONTROL_POLYGON')
        viz = None
        if self.viz:
            viz = self._viz_payload(out, viz_candidates)
        return dict(path=Path.build(request['frame'].world(scene)), controls=controls,
                    corridor=out['corridors'][0], viz=viz,
                    curve_scene=scene, guided=bool(out['guided'][0]), alm_status=out['alm_status'][0],
                    validation=out['final_validation'][0], topology=int(out['selected_idx'][0]),
                    activation=out['activation_info'], timing=out['timing'],
                    ellipse_center=(None if out.get('ellipse_center') is None else
                                    out['ellipse_center'].detach().float().cpu().numpy()),
                    ellipse_ab=(None if out.get('ellipse_a') is None else
                                np.stack([out['ellipse_a'].detach().float().cpu().numpy(),
                                          out['ellipse_b'].detach().float().cpu().numpy()], axis=-1)))

    def _viz_payload(self, out, candidates):
        """Real sampler internals for rendering; never synthesized for the video."""
        import torch
        # Trace tensors are CPU clones while the codec basis follows the model
        # device, so decode explicitly on CPU.
        basis = self.model.bspline.basis.detach().float().cpu()

        def decode(q):
            return torch.einsum('hk,bkd->bhd', basis, q.detach().float().cpu()).numpy()[0]

        last = out['trace'][-1]
        steps = []
        for tr in out['trace']:
            delta = (tr['q0_safe']-tr['q0_raw']).detach().float().cpu().numpy()[0]
            steps.append(dict(t=int(tr['t']), guided=bool(tr['guided'][0]), active=bool(tr['alm_active']),
                              delta=float(np.linalg.norm(delta, axis=1).max()) if delta.size else 0.))
        corridor = out['corridors'][0] or {}
        polygons = [np.asarray(cell['polygon'], dtype=np.float64) for cell in corridor.get('cells', ())
                    if len(cell.get('polygon') or ()) >= 3]
        overlap = [float(v) for v in corridor.get('overlap_ratio', ())]
        return dict(raw_curve=decode(last['q0_raw']), safe_curve=decode(last['q0_safe']),
                    raw_controls=last['q0_raw'].detach().float().cpu().numpy()[0],
                    safe_controls=last['q0_safe'].detach().float().cpu().numpy()[0],
                    steps=steps,
                    corridor=dict(valid=bool(corridor.get('valid')),
                                  cells=len(polygons), overlap_min=min(overlap) if overlap else None,
                                  bridge=int(corridor.get('bridge_cell_count', 0)),
                                  failure=corridor.get('failure_reason'),
                                  polygons=polygons),
                    candidates=candidates, selected=int(out['selected_idx'][0]))
