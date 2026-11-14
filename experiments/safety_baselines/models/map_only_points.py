"""Map/endpoints-only conditional x0 denoiser. No skeleton or safety modules.
Public module implementations are reused; Ours source remains read-only.
"""
import torch
from torch import nn
from src.geometry.ellipse_raster import scene_grid_centres
from src.models.common.scene_cnn import SceneCNN
from src.models.position_encoding import Sinusoidal1DPositionEmbedding,Sinusoidal2DPositionEmbedding,SinusoidalTimestepEmbedding
from src.models.trajsafe.encoders import TrajectoryEncoder
from src.models.trajsafe.blocks import TrajBlock
from src.models.trajsafe.fusion import FusionMLP,FinalDenoiser
from src.models.trajsafe.boundary import BoundaryDecoder

def pin_endpoints(points,cond):
    result=points.clone();result[:,0]=cond[:,0];result[:,-1]=cond[:,1];return result

class MapOnlyPointDiffuser(nn.Module):
    architecture='map_only_point_diffuser_v1'
    def __init__(self,cfg):
        super().__init__();m=cfg['model'];d=m['d_model'];self.num_points=int(cfg['trajectory']['num_points'])
        self.global_res=m['global_mem_res'];self.traj_blocks=m['traj_blocks'];self.final_blocks=m['final_blocks']
        self.scene_cnn=SceneCNN(d,m['map_res'],self.global_res)
        self.spatial_pe=Sinusoidal2DPositionEmbedding(d);self.index_pe=Sinusoidal1DPositionEmbedding(d);self.time_pe=SinusoidalTimestepEmbedding(d)
        self.traj_encoder=TrajectoryEncoder(d,self.spatial_pe,self.index_pe,self.num_points,m.get('coord_hidden',256))
        self.traj_backbone=nn.ModuleList([TrajBlock(d,m['num_heads'],m['ffn_dim'],self.num_points,m.get('dropout',0.),rel_bias_len=m['rel_bias_len']) for _ in range(self.traj_blocks)])
        self.head_p=nn.Linear(d,2)
        self.fusion_mlp=FusionMLP(d,num_streams=1)
        self.final_denoiser=FinalDenoiser(d,m['num_heads'],m['ffn_dim'],self.num_points,blocks=self.final_blocks,dropout=m.get('dropout',0.),rel_bias_len=m['rel_bias_len'])
        self.boundary_decoder=BoundaryDecoder(**m['boundary_decoder'])
    def encode_scene(self,occ):
        mem=self.scene_cnn(occ).flatten(2).transpose(1,2)
        grid=scene_grid_centres(self.global_res,mem.device,dtype=mem.dtype)
        return mem+self.spatial_pe(grid)[None]
    def forward_from_memory(self,q_t,mem,cond,t):
        if q_t.shape[1:]!=(self.num_points,2):raise ValueError("expected direct trajectory points")
        q_t=pin_endpoints(q_t,cond);time=self.time_pe(t)
        h=self.traj_encoder(q_t)
        for blk in self.traj_backbone:h=blk(h,mem,time)
        coarse=self.head_p(h)
        final=self.head_p(self.final_denoiser(self.fusion_mlp(h),mem,time))
        return {'coarse_raw':coarse,'trajectory_raw':final,'trajectory':pin_endpoints(self.boundary_decoder(final,cond),cond)}
    def forward(self,q_t,occ,cond,t):
        return self.forward_from_memory(q_t,self.encode_scene(occ),cond,t)

def load_point_baseline(checkpoint,device='cpu'):
    ck=torch.load(checkpoint,map_location=device,weights_only=False)
    if ck.get('architecture')!=MapOnlyPointDiffuser.architecture:raise ValueError('wrong baseline architecture')
    model=MapOnlyPointDiffuser(ck['config']).to(device)
    model.load_state_dict(ck['model_state'],strict=True)
    return model.eval(),ck
