"""Mock API test: checks Gym + Genesis interaction flow, NOT physical validity."""
from __future__ import annotations
import importlib
import math
import sys
import types
import numpy as np


def test_genesis_gym_api_flow(monkeypatch):
    gym = types.ModuleType('gymnasium')
    class Env:
        def reset(self, seed=None):
            self.np_random=np.random.default_rng(seed)
    class Discrete:
        def __init__(self,n): self.n=n
        def contains(self,a): return isinstance(a,(int,np.integer)) and 0<=a<self.n
    class Box:
        def __init__(self,low,high,shape,dtype): self.shape=shape;self.dtype=dtype
    gym.Env=Env
    gym.spaces=types.SimpleNamespace(Discrete=Discrete,Box=Box)
    monkeypatch.setitem(sys.modules,'gymnasium',gym)

    class FakeJoint:
        def __init__(self,i):self.dofs_idx_local=[i]
    class FakeRobot:
        def __init__(self):
            self.pos=np.array([-1.4,-1.3,0.169]);self.yaw=0.;self.cmd=np.zeros(2);self.n_dofs=8
        def get_joint(self,name):return FakeJoint({'left_wheel_joint':6,'right_wheel_joint':7}[name])
        def set_pos(self,p):self.pos=np.array(p,dtype=float)
        def set_quat(self,q):self.yaw=2*math.atan2(float(q[3]),float(q[0]))
        def get_pos(self):return self.pos
        def get_quat(self):return np.array([math.cos(self.yaw/2),0.,0.,math.sin(self.yaw/2)])
        def set_dofs_velocity(self,*a,**k):self.cmd=np.zeros(2)
        def set_dofs_kp(self,*a,**k):pass
        def set_dofs_kv(self,*a,**k):pass
        def set_dofs_force_range(self,*a,**k):pass
        def control_dofs_velocity(self,velocity,dofs_idx_local=None):self.cmd=np.asarray(velocity)
    class FakeMarker:
        def __init__(self): self.pos=np.array([1.35,1.25,.1])
        def set_qpos(self,p): self.pos=np.asarray(p[:3],dtype=float)
    class FakeScene:
        def __init__(self,**kwargs):self.robot=FakeRobot();self.marker=FakeMarker()
        def add_entity(self,morph,**kwargs):
            return self.robot if morph=='urdf' else self.marker
        def build(self):pass
        def step(self):
            left,right=self.robot.cmd
            v=(left+right)/2*.065
            omega=(right-left)*.065/.270
            self.robot.yaw+=omega*.02
            self.robot.pos[0]+=v*math.cos(self.robot.yaw)*.02
            self.robot.pos[1]+=v*math.sin(self.robot.yaw)*.02
    gs=types.ModuleType('genesis')
    gs.Scene=FakeScene
    gs.options=types.SimpleNamespace(SimOptions=lambda **kw:kw)
    gs.morphs=types.SimpleNamespace(Plane=lambda **kw:'plane',Box=lambda **kw:'box',Sphere=lambda **kw:'sphere',URDF=lambda **kw:'urdf')
    gs.materials=types.SimpleNamespace(Rigid=lambda **kw:'rigid',Kinematic=lambda **kw:'kinematic')
    gs.surfaces=types.SimpleNamespace(Default=lambda **kw:'surface')
    monkeypatch.setitem(sys.modules,'genesis',gs)
    # Make sure it sees mocked Gym on a clean import, do not leak the module.
    monkeypatch.delitem(sys.modules,'scout_genesis.env',raising=False)
    from scout_genesis.env import GenesisOdorEnv
    env=GenesisOdorEnv(seed=11)
    obs,_=env.reset(seed=11,options={'start_xy':(-1.4,-1.3),'source_xy':(1.35,1.25),'yaw':0.})
    assert obs.shape==(15,)
    old_x,_,_=env.pose()
    obs,r,term,trunc,info=env.step(1)
    assert env.pose()[0]>old_x
    assert np.isfinite(obs).all()
    assert np.isfinite(r)
    assert isinstance(term,bool) and isinstance(trunc,bool)
    assert info['source_x']==1.35
    assert 'source_x' not in env.trace[0]  # policy and trace stay separate
    # Restore module import (other tests don't import env)
    monkeypatch.delitem(sys.modules,'scout_genesis.env',raising=False)
