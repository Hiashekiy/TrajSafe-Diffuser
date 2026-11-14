import numpy as np
import torch
from src.utils.config import load_config
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.models.map_only_points import MapOnlyPointDiffuser,load_point_baseline
from experiments.safety_baselines.models.point_data import PointDataset
from experiments.safety_baselines.adapters.point_sampler import sample_points
from experiments.safety_baselines.scripts.train_map_only_points import loss_batch

def config():
    c=load_config('experiments/safety_baselines/configs/map_only_points_l1_final8_e100.json')
    c['model'].update(d_model=32,num_heads=4,ffn_dim=64,coord_hidden=32,map_res=32,global_mem_res=4)
    return c

def test_points_forward_backward_sampling_checkpoint(tmp_path):
    c=config();m=MapOnlyPointDiffuser(c);s=NoiseSchedule(4)
    assert not hasattr(m,'bspline') and not hasattr(m,'num_controls')
    assert m.traj_blocks==1 and m.final_blocks==8
    assert not any('bspline' in k or 'skeleton' in k or 'ellipse' in k for k in m.state_dict())
    cond=torch.tensor([[[-.8,0.],[.8,0.]]]);p=torch.linspace(-.8,.8,128)[None,:,None].expand(-1,-1,2).clone();p[:,:,1]=0
    batch={'pos':p,'occupancy':torch.zeros(1,1,32,32),'cond':cond}
    loss,_,out=loss_batch(m,s,batch,c,'cpu');loss.backward()
    assert torch.isfinite(loss) and m.head_p.weight.grad.norm()>0
    assert out['trajectory'].shape==(1,128,2)
    torch.testing.assert_close(out['trajectory'][:,[0,-1]],cond,atol=0,rtol=0)
    m.eval();a=sample_points(m,s,batch['occupancy'],cond,7,4);b=sample_points(m,s,batch['occupancy'],cond,7,4)
    torch.testing.assert_close(a,b,atol=0,rtol=0)
    torch.testing.assert_close(a[:,[0,-1]],cond,atol=0,rtol=0)
    path=tmp_path/'points.pt';torch.save({'architecture':m.architecture,'config':c,'model_state':m.state_dict()},path)
    loaded,_=load_point_baseline(path)
    torch.testing.assert_close(sample_points(loaded,s,batch['occupancy'],cond,7,4),a,atol=0,rtol=0)

def test_point_dataset_needs_no_control_or_skeleton_files(tmp_path):
    path=tmp_path/'train';path.mkdir()
    points=np.linspace([-.8,0],[.8,0],128,dtype=np.float32)[None]
    np.save(path/'curve_gt.npy',points);np.save(path/'conditions.npy',points[:,[0,-1]])
    np.save(path/'occupancy.npy',np.zeros((1,32,32),np.float32))
    row=PointDataset('train',tmp_path)[0]
    assert set(row)=={'pos','occupancy','cond'} and row['pos'].shape==(128,2)
