import sys, numpy as np, torch
sys.path.insert(0, ".")
from src.utils.config import load_config
from src.datasets.carla_spline_dataset import CarlaSplineDataset, make_collate
from src.utils.checkpoint import load_model
from src.diffusion.schedule import NoiseSchedule
from src.diffusion.sampler import sample

RES = 256
def log(*a):
    print(*a, flush=True)

cfg = load_config("configs/config_160.yaml")
dev = "cuda" if torch.cuda.is_available() else "cpu"
ds = CarlaSplineDataset("val", "data/carla_processed_160", geometry_points=1280,
                        indices=list(range(32, 48)), require_labels=False)
model, _, _ = load_model(cfg, "outputs/bspline_carla_160/ckpt/best_task.pt",
                         arch="auto", device=dev)
model.eval()
sched = NoiseSchedule(cfg["diffusion"]["timesteps"],
                      beta_schedule=cfg["diffusion"].get("beta_schedule",
                                                         "squaredcos_cap_v2")).to(dev)
b = make_collate(ds)([ds[i] for i in range(16)])
with torch.no_grad():
    out = sample(model, sched, b["cond"].to(dev), b["occupancy"].to(dev),
                 b["candidate_xy"].to(dev), b["candidate_mask"].to(dev),
                 b["candidate_geometry"].to(dev),
                 b["candidate_geometry_lengths"].to(dev),
                 device=dev, steps=None, seed=0, return_trace=False,
                 alm_config=dict(cfg.get("alm") or {}),
                 corridor_config=dict(cfg.get("corridor") or {}))
P = out["p"].cpu().numpy(); GT = b["pos"].numpy(); OCC = b["occupancy"].numpy()[:, 0]
SEL = out["selected_idx"].cpu().numpy(); BEST = b["topology_best"].numpy()

k = 3                       # global index 35
occ = OCC[k]; pr = P[k]; gt = GT[k]
px = np.rint((pr[:, 0] + 1) / 2 * RES - 0.5).astype(int)
py = np.rint((pr[:, 1] + 1) / 2 * RES - 0.5).astype(int)
inside = (px >= 0) & (px < RES) & (py >= 0) & (py < RES)
on_obs = np.zeros(len(pr), bool)
on_obs[inside] = occ[py[inside], px[inside]] > 0
outside = ~inside
coll = on_obs | outside

log("=" * 62)
log("sample index 35   m=%d (best=%d)   occ unique=%s" % (SEL[k], BEST[k], np.unique(occ)))
log("curve points            : %d" % len(pr))
log("collision-flagged total : %d  (%.1f%%)" % (coll.sum(), 100 * coll.mean()))
log("   OUTSIDE the 256 grid : %d" % outside.sum())
log("   ON an obstacle cell  : %d" % on_obs.sum())
log("pred scene x range [%.4f, %.4f]  y range [%.4f, %.4f]"
    % (pr[:, 0].min(), pr[:, 0].max(), pr[:, 1].min(), pr[:, 1].max()))
log("GT   scene x range [%.4f, %.4f]  y range [%.4f, %.4f]"
    % (gt[:, 0].min(), gt[:, 0].max(), gt[:, 1].min(), gt[:, 1].max()))
# does the GT pass the same test?
gpx = np.rint((gt[:, 0] + 1) / 2 * RES - 0.5).astype(int)
gpy = np.rint((gt[:, 1] + 1) / 2 * RES - 0.5).astype(int)
gin = (gpx >= 0) & (gpx < RES) & (gpy >= 0) & (gpy < RES)
gobs = np.zeros(len(gt), bool); gobs[gin] = occ[gpy[gin], gpx[gin]] > 0
log("GT collision-flagged    : %d  (outside %d, on-obstacle %d)"
    % ((gobs | ~gin).sum(), (~gin).sum(), gobs.sum()))
log("")
idx = np.nonzero(coll)[0]
log("colliding indices: %s" % idx.tolist())
for i in idx[:20]:
    kind = "OUTSIDE_GRID" if outside[i] else ("OBSTACLE" if on_obs[i] else "free")
    log("   i=%3d scene=(%+.4f,%+.4f) cell=(%4d,%4d) occ=%d  %s"
        % (i, pr[i, 0], pr[i, 1], px[i], py[i],
           occ[py[i], px[i]] if inside[i] else -1, kind))
if len(idx):
    runs, s = [], idx[0]
    for a, bb in zip(idx, idx[1:]):
        if bb != a + 1:
            runs.append((int(s), int(a))); s = bb
    runs.append((int(s), int(idx[-1])))
    log("consecutive runs: %s" % runs)
log("DONE")
