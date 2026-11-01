"""Only tests components not requiring Genesis; physics requires the Colab smoke test."""
import math
import xml.etree.ElementTree as ET
from pathlib import Path
import numpy as np
import pytest
from scout_genesis.core import (
    ACTION_LABELS, OdorField, ROOM_HALF, WHEEL_RADIUS, WHEEL_BASE,
    SOURCE_CANDIDATES, START_CANDIDATES, distance_to_source,
    motor_target, observe, raycast_xy, yaw_from_quat, yaw_quat,
)


def test_odor_source_has_maximum_signal():
    field=OdorField(0.4,0.3)
    assert field.at(.4,.3)==pytest.approx(1.0)
    assert field.at(2,2)<0.1
    assert 0<=field.at(1,1)<=1


def test_yaw_roundtrip():
    for angle in (-math.pi,-1.5,0.,1.2,math.pi):
        got=yaw_from_quat(yaw_quat(angle))
        assert math.sin(got)==pytest.approx(math.sin(angle),abs=1e-6)
        assert math.cos(got)==pytest.approx(math.cos(angle),abs=1e-6)


def test_motor_commands_left_right_convention():
    assert motor_target(1).tolist()==[8,8]
    assert motor_target(2)[0]<0<motor_target(2)[1]
    assert motor_target(3)[1]<0<motor_target(3)[0]
    with pytest.raises(ValueError): motor_target(99)


def test_lidar_boundary_and_obstacle():
    assert raycast_xy(0,-1, -math.pi/2)==pytest.approx(1.)
    assert raycast_xy(0,0,0)==pytest.approx(1.5)
    # From directly below a central-left obstacle, ray points +y.
    assert raycast_xy(-.45,-.5,math.pi/2)==pytest.approx(.45)


def test_observation_is_finite_and_contains_no_source_coordinates():
    arr=observe(0.,0.,0.,.2,.18,2,3,.01,.01)
    assert arr.shape==(15,) and arr.dtype==np.float32
    assert np.isfinite(arr).all()
    assert np.all(arr>=-5) and np.all(arr<=5)


def test_starts_and_sources_are_inside_room():
    for x,y in (*START_CANDIDATES,*SOURCE_CANDIDATES):
        assert abs(x)<ROOM_HALF and abs(y)<ROOM_HALF


def test_urdf_contains_two_actuated_wheels():
    root=ET.parse(Path(__file__).parents[1]/'assets/scout_diffdrive.urdf').getroot()
    joints={j.attrib['name']:j for j in root.findall('joint')}
    assert {'left_wheel_joint','right_wheel_joint'}<=joints.keys()
    for name in ('left_wheel_joint','right_wheel_joint'):
        assert joints[name].attrib['type']=='continuous'
        assert joints[name].find('axis').attrib['xyz']=='0 1 0'
    left=np.fromstring(joints['left_wheel_joint'].find('origin').attrib['xyz'],sep=' ')
    right=np.fromstring(joints['right_wheel_joint'].find('origin').attrib['xyz'],sep=' ')
    assert left[1]-right[1]==pytest.approx(WHEEL_BASE)
    radii=[float(c.attrib['radius']) for c in root.findall('.//cylinder')]
    assert radii and all(r==pytest.approx(WHEEL_RADIUS) for r in radii)
