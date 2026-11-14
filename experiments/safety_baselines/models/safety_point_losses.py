"""Losses for the native-point safety model: the S-C48 recipe, adapted to a point state.

S-C48's recipe (read from `outputs/models/oneshot_k4p_c48_s/ckpt/best_task.pt`):

    L = 0.2  * L_points     (x0 MSE; was L_control on 48 B-spline controls)
      + 0.5  * L_coarse
      + 0.08 * L_smooth     (2nd difference, acc 0.25 / jerk 1.0; was on the control polygon)
      + 0.2  * L_boundary   (endpoint error, coarse weighted 0.5)
      + 0.25 * L_topology   (candidate CE vs topology_best)
      + 0.08 * L_shape      (ellipse 4-parameter MSE vs ellipse_shape4_gt)
      + 0.25 * L_iou        (soft IoU of the predicted vs GT ellipse raster)
      + 0.15 * L_safe       (Mean + CVaR of the unsafe-area fraction)
      + 0.3  * L_alm        (penetration of the DELIVERED curve into the corridor pack)

What the point state changes (and nothing else):

  * `L_points` / `L_coarse` act on the 128 delivered points; there is no `B Q` decode.
  * `L_smooth` uses the INDEX parameter of the point sequence. On the control polygon the
    second difference measures the polygon, not the curve, so the control recipe is not
    transferable; on points it is the standard discrete-curvature proxy, which is the term
    this model needs. No arc-length re-weighting: `curve_gt` is already arc-length
    parameterised, so the index is proportional to arc length.
  * `L_alm` receives the points directly (the same object the corridor constrains).
  * `L_shape` / `L_iou` / `L_safe` are representation-free: they live on the Q=128 ellipse
    anchors placed along the selected candidate. `shape4 = [log a, log b, cos2t, sin2t]`
    carries NO centre, so both the prediction and the GT mask are rasterised at the same
    anchor centres -- no extra dataset array is needed.
"""
import torch
import torch.nn.functional as F

from src.geometry.ellipse_raster import ellipse_soft_mask, occupancy_raster
from src.geometry.ellipse_shape import shape4_to_abtheta


# ---------------------------------------------------------------- point terms
def point_x0_loss(pred, gt, mask=None, eps=1e-6):
    err = ((pred - gt) ** 2).sum(-1)
    if mask is None:
        return err.mean()
    m = mask.to(err.dtype).reshape(-1, 1).expand_as(err)
    return (err * m).sum() / (m.sum() + eps)


def point_smoothness_loss(pred, gt, acc_weight=0.25, jerk_weight=1.0, mask=None):
    d2 = torch.diff(pred, n=2, dim=1) - torch.diff(gt, n=2, dim=1)
    d3 = torch.diff(pred, n=3, dim=1) - torch.diff(gt, n=3, dim=1)
    acc = (d2 ** 2).sum(-1); jerk = (d3 ** 2).sum(-1)
    if mask is not None:
        m2 = mask.to(acc.dtype).reshape(-1, 1).expand_as(acc)
        m3 = mask.to(jerk.dtype).reshape(-1, 1).expand_as(jerk)
        acc = (acc * m2).sum() / m2.sum().clamp_min(1.0)
        jerk = (jerk * m3).sum() / m3.sum().clamp_min(1.0)
    else:
        acc = acc.mean(); jerk = jerk.mean()
    return acc_weight * acc + jerk_weight * jerk


def point_boundary_loss(pred, coarse, cond, coarse_weight=0.5):
    e_f = ((pred[:, 0] - cond[:, 0]) ** 2).sum(-1) + ((pred[:, -1] - cond[:, 1]) ** 2).sum(-1)
    e_c = ((coarse[:, 0] - cond[:, 0]) ** 2).sum(-1) + ((coarse[:, -1] - cond[:, 1]) ** 2).sum(-1)
    return 0.5 * (e_f.mean() + coarse_weight * e_c.mean())


# ---------------------------------------------------------------- ellipse terms
def ellipse_shape_loss(shape4, shape4_gt, shape_valid, sample_mask):
    err = ((shape4 - shape4_gt) ** 2).sum(-1)
    m = shape_valid.to(err.dtype) * sample_mask.to(err.dtype).reshape(-1, 1)
    return (err * m).sum() / (m.sum() + 1e-6)


def ellipse_iou_loss(center, a, b, theta, shape4_gt, shape_valid, sample_mask,
                     raster_res=64, tau=10.0):
    """1 - soft IoU, with BOTH masks rasterised at the same anchor centres."""
    pred_mask = ellipse_soft_mask(center, a, b, theta, raster_res, tau)
    gt_a, gt_b, gt_theta = shape4_to_abtheta(shape4_gt)
    gt_mask = ellipse_soft_mask(center, gt_a, gt_b, gt_theta, raster_res, tau)
    inter = (pred_mask * gt_mask).flatten(2).sum(-1)
    union = (pred_mask.flatten(2).sum(-1) + gt_mask.flatten(2).sum(-1) - inter).clamp_min(1e-6)
    iou = inter / union
    m = shape_valid.to(iou.dtype) * sample_mask.to(iou.dtype).reshape(-1, 1)
    return ((1.0 - iou) * m).sum() / (m.sum() + 1e-6)


def ellipse_safety_loss(center, a, b, theta, occ, raster_res=64, tau=10.0,
                        cvar_fraction=0.2, cvar_weight=1.0, sample_mask=None, eps=1e-6):
    """Mean + CVaR of the unsafe-area fraction of the full ellipses (S-C48 definition)."""
    # occupancy_raster ALREADY returns the free mask (1 = free), so it is not inverted.
    free = occupancy_raster(occ.float(), raster_res).float()            # [B,1,r,r]
    pred = ellipse_soft_mask(center, a, b, theta, raster_res, tau)      # [B,Q,r,r]
    inside = (pred * free).flatten(2).sum(-1)
    total = pred.flatten(2).sum(-1).clamp_min(eps)
    # u is a FRACTION and must stay in [0,1]; clamping also guards the tiny-mask regime
    # (very small predicted ellipses) where the ratio is numerically meaningless.
    u = (1.0 - inside / total).clamp(0.0, 1.0)                          # [B,Q]
    q = u.shape[-1]; k = max(1, int(round(cvar_fraction * q)))
    if sample_mask is not None:
        m = sample_mask.to(u.dtype).reshape(-1, 1)
        # the numerator sums over BOTH batch and queries, so the denominator must too
        mean = (u * m).sum() / (m.sum() * u.shape[-1]).clamp_min(1.0)
        top = torch.topk(u * m, k, dim=-1).values
        cnt = m.expand_as(top).sum(-1).clamp_min(1.0)
        cvar = (top.sum(-1) / cnt).mean()
    else:
        mean = u.mean(); cvar = torch.topk(u, k, dim=-1).values.mean()
    return mean + cvar_weight * cvar


# ---------------------------------------------------------------- corridor term
def alm_corridor_loss(points, cell_a, cell_b, cell_valid, sample_mask, margin=0.0):
    """Penetration of the DELIVERED points into the corridor (S-C48 semantics).

    The corridor is a UNION of cells: ``p`` is inside cell ``i`` iff ``A_if . p <= b_if``
    for every face ``f``. With unit-normalised rows,

        violation(p) = min_i max_f (A_if . p - b_if)

    is positive exactly when ``p`` is outside EVERY cell (outside the corridor), and its
    value is the distance to the nearest cell boundary in scene units. Padding faces carry
    ``b = +inf`` so they can never bind (``A.p - inf = -inf`` never wins the inner max),
    and cells flagged invalid are masked to ``+inf`` so they never win the outer min.

    The control-space version decoded ``B Q`` into points first; here the points ARE the
    curve. Shapes: points [B,H,2], cell_a [B,C,F,2], cell_b [B,C,F], cell_valid [B,C].
    """
    if cell_a.dim() != 4:
        raise ValueError('cell_a must be [B,C,F,2], got %s' % (tuple(cell_a.shape),))
    n_cells = cell_a.shape[1]
    if cell_valid.shape != (cell_a.shape[0], n_cells):
        raise ValueError('cell_valid %s vs cells %d' % (tuple(cell_valid.shape), n_cells))
    d = (cell_a[:, None] * points[:, :, None, None, :]).sum(-1) - cell_b[:, None]  # [B,H,C,F]
    out_of_cell = d.amax(dim=-1)                                             # [B,H,C]
    out_of_cell = torch.where(cell_valid[:, None].bool(),
                              out_of_cell, torch.full_like(out_of_cell, float('inf')))
    violation = out_of_cell.amin(dim=-1)                                     # [B,H]
    violation = torch.nan_to_num(violation, posinf=0.0)
    pen = torch.clamp(violation + margin, min=0.0)
    m = sample_mask.to(pen.dtype).reshape(-1, 1).expand_as(pen)
    return (pen * m).sum() / m.sum().clamp_min(1.0)


# ---------------------------------------------------------------- composition
def compose_losses(loss_cfg, out, batch, cond, occ, sample_mask=None):
    """One batch of the recipe above. Returns (total, {term: value})."""
    lam = loss_cfg
    gt = batch["curve_gt"]
    # No per-sample validity array exists in this cache, so a full mask is the correct
    # default; it stays an argument so partial batches / padding can be masked later.
    m = sample_mask if sample_mask is not None else torch.ones(gt.shape[0], dtype=torch.bool,
                                                               device=gt.device)
    parts = {}
    pred, coarse = out["points"], out["coarse"]
    parts["points"] = point_x0_loss(pred, gt, m)
    parts["coarse"] = point_x0_loss(coarse, gt, m)
    parts["smooth"] = point_smoothness_loss(pred, gt, lam.get("smooth_acc_weight", 0.25),
                                            lam.get("smooth_jerk_weight", 1.0), m)
    parts["boundary"] = point_boundary_loss(pred, coarse, cond, lam.get("boundary_coarse_weight", 0.5))
    logits = out["topology_logits"].clone()
    logits[~batch["candidate_mask"]] = float("-inf")
    best = batch["topology_best"]
    valid_best = batch["candidate_mask"].gather(1, best.clamp_min(0).reshape(-1, 1)).reshape(-1)
    best = torch.where(valid_best, best, batch["candidate_mask"].float().argmax(-1))
    parts["topology"] = F.cross_entropy(logits, best)

    shape = out.get("shape4")
    if shape is not None and "shape4_gt" in batch:
        center = out["center"]
        parts["shape"] = ellipse_shape_loss(shape["shape4"], batch["shape4_gt"],
                                            batch["shape_valid"], m)
        parts["iou"] = ellipse_iou_loss(center, shape["a"], shape["b"], shape["theta"],
                                        batch["shape4_gt"], batch["shape_valid"], m,
                                        raster_res=int(lam.get("ellipse_iou_res", 64)),
                                        tau=float(lam.get("ellipse_mask_tau", 10.0)))
        parts["safe"] = ellipse_safety_loss(center, shape["a"], shape["b"], shape["theta"], occ,
                                            raster_res=int(lam.get("ellipse_safe_res", 64)),
                                            tau=float(lam.get("ellipse_mask_tau", 10.0)),
                                            cvar_fraction=float(lam.get("safe_cvar_fraction", 0.2)),
                                            cvar_weight=float(lam.get("safe_cvar_weight", 1.0)),
                                            sample_mask=m)
        parts["alm"] = alm_corridor_loss(pred, batch["cell_a"], batch["cell_b"],
                                         batch["cell_valid"], m)

    total = (lam["lambda_points"] * parts["points"]
             + lam["lambda_coarse"] * parts["coarse"]
             + lam["lambda_smooth"] * parts["smooth"]
             + lam["lambda_boundary"] * parts["boundary"]
             + lam["lambda_topology"] * parts["topology"])
    for key, weight in (("shape", "lambda_shape"), ("iou", "lambda_iou"),
                        ("safe", "lambda_safe"), ("alm", "lambda_alm")):
        if key in parts:
            total = total + lam[weight] * parts[key]
    return total, {k: float(v.detach().float()) for k, v in parts.items()}
