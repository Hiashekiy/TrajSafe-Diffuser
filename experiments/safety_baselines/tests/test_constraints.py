import numpy as np
from experiments.safety_baselines.adapters.constraints import MapConstraint

def test_chain_rule_and_boundary():
    occ=np.zeros((16,16),bool);occ[6:10,6:10]=True
    c=MapConstraint(occ,extent=16)
    Q=np.array([[-5.2,-3.1],[4.3,5.2]])
    B=np.array([[.7,.3],[.25,.75]])
    g,J=c.control_constraints(Q,B)
    for k in range(Q.size):
        delta=np.zeros(Q.size);delta[k]=1e-6
        plus=c.control_constraints(Q+delta.reshape(Q.shape),B)[0]
        minus=c.control_constraints(Q-delta.reshape(Q.shape),B)[0]
        np.testing.assert_allclose((plus-minus)/2e-6,J[:,k],atol=1e-7)
    h,_=c.query(np.array([[9.,0.],[0.,0.],[-6.,-6.]]))
    assert h[0]<0 and h[1]<0 and h[2]>0

def test_empty_and_full():
    assert MapConstraint(np.zeros((8,8)),8).query(np.array([[0.,0.]]))[0][0]>0
    assert MapConstraint(np.ones((8,8)),8).query(np.array([[0.,0.]]))[0][0]<0
