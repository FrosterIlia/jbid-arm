import math, time
import numpy as np
import pinocchio as pin
from robstride_dynamics import RobstrideBus, Motor, ParameterType
from constants import *
from ArmHardwareInterface import ArmHardware
from kinematics import KinematicsSolver
from CameraStream import CameraStream
from PositionEstimator import PositionEstimator
from debug_view import DebugView

arm = ArmHardware("pcan0")
kinematics = KinematicsSolver("urdf/ReBot_Arm_RS.urdf")
camera = CameraStream(0, 640, 480, 30)
position_estimator = PositionEstimator(0.42)

arm.enable_all()

HOME_POSITION = [ 0.3006, -0.0201,  0.2197]
HOME_ROTATION = [[0.992,  0.121, -0.021], [-0.121,  0.992, -0.029],  [0.017,  0.032,  0.999]]

TARGET_POSITION = [0.3231, 0.0021, 0.2772]
TARGET_ROTATION = [[1, 0.01,  -0.024], [-0.011,  0.999, -0.03], [0.023,  0.031,  0.999]]


# Gravity feedforward per joint (j1..j6). 1.0 = full URDF estimate, 0.0 = off, -1.0 = flip the sign
# if the motor's positive direction is opposite to the URDF joint axis. Start low and raise it.
GRAVITY_SCALE = np.array([0.0, 0.0, 0.0, 1.45, 0.0, 0.0])


def log_state(joint_states):
    line = "  ".join(f"{n}: {p:+.3f} ({v:+.2f})" for n, (p, v) in joint_states.items())
    q = np.array([joint_states[n][0] for n in arm.names[:kinematics.N_JOINTS]])
    fk = kinematics.forward_kinematics(q)
    # print(f"\r\033[K{line}", end='', flush=True)
    rpy = pin.rpy.matrixToRpy(fk.rotation)
    rot = np.array2string(fk.rotation.round(3).ravel(), max_line_width=1000)
    print(f"\r\033[Kxyz = {fk.translation.round(4)} m   rpy = {rpy.round(3)} rad, rot = {rot}", end='', flush=True)


CAM_TO_EE = np.array([[0.0, 0.0, 1.0], [-1.0, 0.0, 0.0], [0.0, -1.0, 0.0]])
GAIN = 0.3
MAX_DRIFT = 0.15
DEAD_ZONE = 0.08     # m, the arm holds still while the marker is this close to MARKER_REF
LOST_TIMEOUT = 0.5   # s without the marker before going home

MARKER_REF = np.array([0, 0, 0.1])

target_pos = np.array(TARGET_POSITION, dtype=float)
target_rot = np.array(TARGET_ROTATION)

debug = DebugView(position_estimator, CAM_TO_EE, MARKER_REF, TARGET_POSITION, MAX_DRIFT, GAIN, arm,
                  dead_zone=DEAD_ZONE, lost_timeout=LOST_TIMEOUT)
last_seen = time.monotonic()

try:
    while True:
        camera.update()
        frame = camera.read()
        marker_pos = position_estimator.estimate(frame, 1)
        joint_states = arm.get_joint_states()
        # log_state(joint_states)

        q = np.array([joint_states[n][0] for n in arm.names[:kinematics.N_JOINTS]])
        fk = kinematics.forward_kinematics(q)

        if marker_pos is not None:
            last_seen = time.monotonic()
            error = marker_pos - MARKER_REF
            if np.linalg.norm(error) > DEAD_ZONE:
                err_base = fk.rotation @ CAM_TO_EE @ error

                target_pos = target_pos + GAIN * err_base
                target_pos = np.clip(target_pos,
                                     np.array(TARGET_POSITION) - MAX_DRIFT,
                                     np.array(TARGET_POSITION) + MAX_DRIFT)
        elif time.monotonic() - last_seen > LOST_TIMEOUT:
            target_pos = np.array(TARGET_POSITION, dtype=float)

        target = pin.SE3(target_rot, target_pos)
        q_ik, ok = kinematics.inverse_kinematics(target, q0=q)
        if not debug.update(frame, marker_pos, q, fk, target_pos, q_ik, ok):
            break
        if not ok:
            print("\nik failed")
            continue
        ik = np.append(q_ik, 0.0)  # j7 (gripper) = 0

        gravity = np.append(kinematics.gravity_torques(q) * GRAVITY_SCALE, 0.0)  # j7 (gripper): none

        arm.set_joint_targets(ik, torques=gravity)


finally:
    debug.close()
    arm.bus.disconnect()