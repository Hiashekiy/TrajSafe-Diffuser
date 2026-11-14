import torch
from src.diffusion.schedule import NoiseSchedule
from experiments.safety_baselines.adapters.baseline_sampler import sample_baseline
class Toy(torch.nn.Module):
    num_controls=48
    def forward(self,q,occ,cond,t):return {'control':q*.1}
def test_plain_ddim_reproducibility_and_endpoints():
    m=Toy();s=NoiseSchedule(4);occ=torch.zeros(2,1,8,8);cond=torch.tensor([[[-.8,0.],[.8,0.]]]*2)
    a=sample_baseline(m,s,occ,cond,7,0,4);b=sample_baseline(m,s,occ,cond,7,0,4)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    torch.testing.assert_close(a[:,0],cond[:,0]);torch.testing.assert_close(a[:,-1],cond[:,1])
    c=sample_baseline(m,s,occ,cond,8,0,4)
    assert not torch.equal(a[:,1:-1],c[:,1:-1]);assert torch.isfinite(a).all()
