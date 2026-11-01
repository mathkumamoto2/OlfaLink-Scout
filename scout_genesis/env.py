"""Gymnasium environment backed by Genesis World 1.4 rigid-body physics.

- Wheel motion and collisions: Genesis's rigid solver.
- Smell: analytical proxy signal, NOT a gas/CFD simulation.
- Obstacles sensing: analytical 2D rays, NOT the Genesis LiDAR sensor.
"""
from __future__ import annotations
import math
from pathlib import Path
from typing import Any

import gymnasium as gym
from gymnasium import spaces
import numpy as np
from .core import (
    ACTION_LABELS, BASE_Z, MAX_STEPS, OBSTACLES, ROOM_HALF, ROBOT_RADIUS,
    SIM_DT, SOURCE_CANDIDATES, START_CANDIDATES, STEPS_PER_ACTION,
    OdorField, distance_to_source, motor_target, observe, raycast_xy,
    yaw_from_quat, yaw_quat,
)

ASSET_FILE = Path(__file__).resolve().parent.parent / 'assets' / 'scout_diffdrive.urdf'


class GenesisOdorEnv(gym.Env):
    """A single robot and one simulated world, no privileged source position in observations."""

    metadata = {'render_modes': []}

    def __init__(self, *, backend: str = 'cpu', render_camera: bool = False,
                 show_viewer: bool = False, seed: int = 42):
        super().__init__()
        try:
            import genesis as gs
        except ImportError as exc:
            raise RuntimeError('Genesis is not installed: pip install -r requirements-genesis.txt') from exc
        self.gs = gs
        self.action_space = spaces.Discrete(len(ACTION_LABELS))
        self.observation_space = spaces.Box(low=-5.0, high=5.0, shape=(15,), dtype=np.float32)
        self._rng = np.random.default_rng(seed)
        self._seed = int(seed)
        self._init_scene(render_camera=render_camera, show_viewer=show_viewer)
        self.steps = 0
        self.source_xy = (1.35, 1.25)
        self.field = OdorField(*self.source_xy)
        self._smell = 0.0
        self._prev_smell = 0.0
        self._last_action = 0
        self._vel = 0.0
        self._omega = 0.0
        self._prev_distance = 0.0
        self._trace: list[dict[str,float|int]] = []

    def _init_scene(self, *, render_camera:bool, show_viewer:bool) -> None:
        gs=self.gs
        self.scene=gs.Scene(
            sim_options=gs.options.SimOptions(dt=SIM_DT, substeps=4),
            show_viewer=show_viewer,
        )
        self.scene.add_entity(gs.morphs.Plane())
        # Static walls: interior surface at x,y=+/-ROOM_HALF.
        for x,y,sx,sy in (
            (0., ROOM_HALF+0.08, 2*ROOM_HALF+0.32, 0.16),
            (0., -ROOM_HALF-0.08, 2*ROOM_HALF+0.32, 0.16),
            (ROOM_HALF+0.08, 0., 0.16, 2*ROOM_HALF),
            (-ROOM_HALF-0.08, 0., 0.16, 2*ROOM_HALF),
        ):
            self.scene.add_entity(
                gs.morphs.Box(pos=(x,y,0.25), size=(sx,sy,0.50), fixed=True),
            )
        for x,y,sx,sy in OBSTACLES:
            self.scene.add_entity(
                gs.morphs.Box(pos=(x,y,0.25), size=(sx,sy,0.50), fixed=True),
            )
        # Marker is visualization-only. The agent never observes its position;
        # it does NOT interfere with Genesis rigid-body collision or learning.
        self.source_marker=self.scene.add_entity(
            gs.morphs.Sphere(pos=(1.35, 1.25, 0.10), radius=0.08),
            material=gs.materials.Kinematic(),
            surface=gs.surfaces.Default(color=(1.0, 0.32, 0.08, 1.0)),
        )
        self.robot=self.scene.add_entity(
            gs.morphs.URDF(file=str(ASSET_FILE), pos=(-1.4,-1.3,BASE_Z), fixed=False),
            material=gs.materials.Rigid(friction=1.0),
        )
        self.camera = None
        if render_camera:
            self.camera=self.scene.add_camera(
                res=(640,480), pos=(3.8,-4.8,5.2), lookat=(0.,0.,0.1),
                fov=55, GUI=False,
            )
        self.scene.build()
        self.motor_indices=[
            int(self.robot.get_joint(name).dofs_idx_local[0])
            for name in ('left_wheel_joint','right_wheel_joint')
        ]
        self.robot.set_dofs_kp(np.zeros(2,dtype=np.float32), dofs_idx_local=self.motor_indices)
        self.robot.set_dofs_kv(np.asarray([15.,15.],dtype=np.float32), dofs_idx_local=self.motor_indices)
        self.robot.set_dofs_force_range(
            np.asarray([-5.,-5.],dtype=np.float32),
            np.asarray([5.,5.],dtype=np.float32),
            dofs_idx_local=self.motor_indices,
        )

    @staticmethod
    def _np(tensor: Any) -> np.ndarray:
        return np.asarray(tensor.detach().cpu().numpy() if hasattr(tensor,'detach') else tensor)

    def pose(self) -> tuple[float,float,float]:
        pos=self._np(self.robot.get_pos()).reshape(-1)
        quat=self._np(self.robot.get_quat()).reshape(-1)
        return float(pos[0]),float(pos[1]),yaw_from_quat(quat)

    def _observation(self) -> np.ndarray:
        x,y,yaw=self.pose()
        return observe(x,y,yaw,self._smell,self._prev_smell,
                       self._last_action,self.steps,self._vel,self._omega)

    def reset(self, *, seed:int|None=None, options:dict|None=None):
        super().reset(seed=seed)
        if seed is not None:
            self._rng=np.random.default_rng(seed)
        # Spawn away from the ground-plane obstacles and randomize source.
        start=START_CANDIDATES[int(self._rng.integers(len(START_CANDIDATES)))]
        valid_sources=[xy for xy in SOURCE_CANDIDATES if distance_to_source(*start,xy)>1.0]
        self.source_xy=valid_sources[int(self._rng.integers(len(valid_sources)))]
        yaw=float(self._rng.uniform(-math.pi,math.pi))
        if options:
            # Deterministic motor/physics diagnostics; training uses randomized resets.
            start=tuple(float(v) for v in options.get('start_xy',start))
            self.source_xy=tuple(float(v) for v in options.get('source_xy',self.source_xy))
            yaw=float(options.get('yaw',yaw))
            if (max(abs(v) for v in start)>ROOM_HALF-ROBOT_RADIUS or
                    distance_to_source(*start,self.source_xy)<0.5):
                raise ValueError('Invalid start_xy/source_xy in reset options')
        self.robot.set_pos(np.asarray((*start,BASE_Z),dtype=np.float32))
        self.robot.set_quat(yaw_quat(yaw))
        # Set the complete generalized velocity to zero. Otherwise, the next
        # episode can inherit rigid-body or wheel inertia from the last one.
        self.robot.set_dofs_velocity(
            np.zeros(self.robot.n_dofs, dtype=np.float32),
            dofs_idx_local=np.arange(self.robot.n_dofs),
        )
        # KinematicEntity exposes set_qpos (not the rigid-only set_pos API).
        # A free primitive uses position xyz + scalar-first quaternion wxyz.
        self.source_marker.set_qpos(
            np.asarray((*self.source_xy, 0.10, 1.0, 0.0, 0.0, 0.0), dtype=np.float32)
        )
        self.robot.control_dofs_velocity(np.zeros(2,dtype=np.float32), self.motor_indices)
        self.field=OdorField(*self.source_xy, wind_angle=float(self._rng.uniform(-0.7,0.7)))
        self.steps=0
        self._smell=self.field.at(*start)
        self._prev_smell=self._smell
        self._last_action=0
        self._vel=0.0
        self._omega=0.0
        self._prev_distance=distance_to_source(*start,self.source_xy)
        self._trace=[]
        return self._observation(), {'source_hidden_from_policy':True}

    def step(self, action):
        action=int(action)
        if not self.action_space.contains(action):
            raise ValueError(f'Action {action} outside {self.action_space}')
        old_x,old_y,old_yaw=self.pose()
        wheel_targets=motor_target(action)
        self.robot.control_dofs_velocity(wheel_targets, dofs_idx_local=self.motor_indices)
        for _ in range(STEPS_PER_ACTION):
            self.scene.step()
        x,y,yaw=self.pose()
        dt=SIM_DT*STEPS_PER_ACTION
        travel=math.hypot(x-old_x,y-old_y)
        dyaw=(yaw-old_yaw+math.pi)%(2*math.pi)-math.pi
        self._vel=float(np.clip((math.cos(old_yaw)*(x-old_x)+math.sin(old_yaw)*(y-old_y))/dt,-1.0,1.0))
        self._omega=dyaw/dt
        self._prev_smell=self._smell
        measured=float(np.clip(self.field.at(x,y)+self._rng.normal(0,0.01),0.,1.))
        self._smell=0.65*self._smell+0.35*measured  # first-order sensor response
        self._last_action=action
        self.steps+=1
        distance=distance_to_source(x,y,self.source_xy)
        success=distance<=0.32
        out_of_bounds=abs(x)>ROOM_HALF-0.04 or abs(y)>ROOM_HALF-0.04 or not np.isfinite(distance)
        # Privileged distance ONLY affects training reward/metrics, not observation.
        reward=4.*(self._prev_distance-distance)+0.5*(self._smell-self._prev_smell)-0.015
        if action == 1 and travel < 0.002 and raycast_xy(x,y,yaw)<ROBOT_RADIUS+0.1:
            reward-=0.20  # proxy for stall against collision geometry
        if success:
            reward+=20.0
        if out_of_bounds:
            reward-=5.0
        self._prev_distance=distance
        self._trace.append({'step':self.steps,'x':x,'y':y,'yaw':yaw,'smell':self._smell,
                            'distance_to_source':distance,'action':action,'reward':float(reward)})
        terminated=bool(success or out_of_bounds)
        truncated=bool(self.steps>=MAX_STEPS and not terminated)
        info={'success':bool(success), 'source_x':self.source_xy[0],
              'source_y':self.source_xy[1], 'distance_to_source':distance,
              'steps':self.steps,'robot_displacement_m':travel}
        obs=self._observation()
        if not np.isfinite(obs).all():
            raise FloatingPointError('Invalid rigid-body state: check robot model/contact and timestep.')
        return obs, float(reward), terminated, truncated, info

    def start_video(self, output_path:Path, fps:int=25) -> None:
        if self.camera is None:
            raise RuntimeError('Camera was not requested; use render_camera=True')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        self.camera.start_recording(save_to_filename=str(output_path.resolve()), fps=fps)

    def stop_video(self) -> None:
        if self.camera is not None:
            self.camera.stop_recording()

    @property
    def trace(self):
        return list(self._trace)

    def close(self):
        # The CLI runs a single Genesis scene per process. Genesis releases
        # its native allocations when the process exits.
        pass
