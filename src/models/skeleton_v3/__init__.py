"""V3 model package: Skeleton-Topology-Grounded Trajectory Diffusion."""

from .blocks import CrossAttention, JointFusionBlock, TrajBlock, TrajSelfAttention
from .ellipse_head import EllipseHead
from .geometry import dense_arclength, gather_dense_path_points
from .path_encoder import StaticPathEncoder
from .planner import SkeletonPlannerV3
from .progress_head import ProgressHead
from .topology_selector import TopologySelector, chamfer_mean_distance

__all__ = [
    "SkeletonPlannerV3",
    "StaticPathEncoder",
    "TopologySelector",
    "chamfer_mean_distance",
    "ProgressHead",
    "EllipseHead",
    "TrajBlock",
    "JointFusionBlock",
    "TrajSelfAttention",
    "CrossAttention",
    "gather_dense_path_points",
    "dense_arclength",
]
