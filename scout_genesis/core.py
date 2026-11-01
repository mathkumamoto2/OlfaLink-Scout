"""Pure NumPy observation, odor and geometry functions (no Genesis dependency)."""
from __future__ import annotations
from dataclasses import dataclass
import math
import numpy as np

ROOM_HALF = 2.0
WHEEL_RADIUS = 0.065
WHEEL_BASE = 0.270  # wheel center separation (meters)
BASE_Z = 0.169
SIM_DT = 0.02
STEPS_PER_ACTION = 8
ROBOT_RADIUS = 0.18
MAX_RAY = 1.5
MAX_STEPS = 160
# XY center and XY size of obstacle boxes. Units: m.
OBSTACLES = ((-0.45, 0.30, 0.40, 0.70), (0.65, -0.50, 0.50, 0.45))
SOURCE_CANDIDATES = ((1.35, 1.25), (1.45, -1.20), (-1.30, 1.20))
START_CANDIDATES = ((-1.40, -1.30), (-1.45, 0.00), (0.0, -1.45))
ACTION_LABELS = ('stop', 'forward', 'turn_left', 'turn_right', 'reverse')

@dataclass(frozen=True)
class OdorField:
    """Dimensionless Gaussian proxy, NOT fluid dynamics or ppm."""
    source_x: float
    source_y: float
    wind_angle: float = 0.5
    sigma_along: float = 0.90
    sigma_cross: float = 0.55

    def at(self, x: float, y: float) -> float:
        dx, dy = float(x)-self.source_x, float(y)-self.source_y
        c, s = math.cos(self.wind_angle), math.sin(self.wind_angle)
        along, across = c * dx + s * dy, -s * dx + c * dy
        return float(math.exp(-0.5 * ((along/self.sigma_along)**2 + (across/self.sigma_cross)**2)))


def yaw_from_quat(q: np.ndarray) -> float:
    """Genesis uses scalar-first quaternion [w,x,y,z]."""
    w, x, y, z = (float(v) for v in np.asarray(q).reshape(-1)[:4])
    return math.atan2(2*(w*z+x*y), 1-2*(y*y+z*z))


def yaw_quat(yaw: float) -> np.ndarray:
    return np.asarray([math.cos(yaw/2), 0., 0., math.sin(yaw/2)], dtype=np.float32)


def motor_target(action: int) -> np.ndarray:
    """Left/right angular velocity commands (rad/s), positive drives +base X."""
    targets = (
        (0., 0.),
        (8., 8.),
        (-5., 5.),
        (5., -5.),
        (-5., -5.),
    )
    if not 0 <= int(action) < len(targets):
        raise ValueError(f'Invalid action {action!r}')
    return np.asarray(targets[int(action)], dtype=np.float32)


def raycast_xy(x: float, y: float, angle: float, max_range: float=MAX_RAY) -> float:
    """Analytic 2D geometry range sensing (not Genesis LiDAR)."""
    dx, dy = math.cos(angle), math.sin(angle)
    boxes = [(-ROOM_HALF, -ROOM_HALF, 2*ROOM_HALF, 2*ROOM_HALF)]
    # Distance to the room boundary (from inside only).
    distances = [float(max_range)]
    if abs(dx)>1e-9:
        for xx in (-ROOM_HALF, ROOM_HALF):
            t = (xx-x)/dx
            yy = y+t*dy
            if 0<=t<=max_range and -ROOM_HALF<=yy<=ROOM_HALF:
                distances.append(t)
    if abs(dy)>1e-9:
        for yy in (-ROOM_HALF, ROOM_HALF):
            t = (yy-y)/dy
            xx = x+t*dx
            if 0<=t<=max_range and -ROOM_HALF<=xx<=ROOM_HALF:
                distances.append(t)
    for cx, cy, sx, sy in OBSTACLES:
        xmin, xmax = cx-sx/2, cx+sx/2
        ymin, ymax = cy-sy/2, cy+sy/2
        # Slab ray/AABB intersection, robust for axis-aligned rays.
        tlo, thi = 0., max_range
        for p, d, lo, hi in ((x,dx,xmin,xmax),(y,dy,ymin,ymax)):
            if abs(d)<1e-9:
                if not lo <= p <= hi:
                    tlo = float('inf')
                    break
            else:
                t1,t2 = sorted(((lo-p)/d,(hi-p)/d))
                tlo,thi = max(tlo,t1),min(thi,t2)
        if tlo<=thi:
            distances.append(tlo)
    return float(min(distances))


def observe(
    x: float, y: float, yaw: float, odor: float, previous_odor: float,
    last_action: int, steps: int, forward_speed: float, turn_speed: float,
) -> np.ndarray:
    """15-dimensional agent observation; excludes source coordinates."""
    rays=[raycast_xy(x,y,yaw+offset)/MAX_RAY for offset in (-math.pi/2,-math.pi/4,0,math.pi/4,math.pi/2)]
    values=[x/ROOM_HALF,y/ROOM_HALF,math.cos(yaw),math.sin(yaw),
            odor, np.clip(odor-previous_odor,-1,1), *rays,
            last_action/(len(ACTION_LABELS)-1), steps/MAX_STEPS,
            float(np.clip(forward_speed,-1,1)),float(np.clip(turn_speed/3.,-1,1))]
    out=np.asarray(values,dtype=np.float32)
    assert out.shape==(15,) and np.isfinite(out).all()
    return out


def distance_to_source(x:float,y:float,source:tuple[float,float]) -> float:
    return math.hypot(x-source[0], y-source[1])
