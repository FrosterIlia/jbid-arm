import math, time
import numpy as np
import pinocchio as pin
from robstride_dynamics import RobstrideBus, Motor, ParameterType
from constants import *
from ArmHardwareInterface import ArmHardware
from kinematics import KinematicsSolver
from CameraStream import CameraStream
from PositionEstimator import PositionEstimator

arm = ArmHardware("pcan0")
kinematics = KinematicsSolver("urdf/ReBot_Arm_RS.urdf")
camera = CameraStream(0, 640, 480, 30)
position_estimator = PositionEstimator(0.42)

arm.enable_all()

HOME_POSITION = [ 0.3006, -0.0201,  0.2197]
HOME_ROTATION = [[0.992,  0.121, -0.021], [-0.121,  0.992, -0.029],  [0.017,  0.032,  0.999]]

TARGET_POSITION = [0.3231, 0.0021, 0.2772]
TARGET_ROTATION = [[1, 0.01,  -0.024], [-0.011,  0.999, -0.03], [0.023,  0.031,  0.999]]



def log_state(joint_states):
    line = "  ".join(f"{n}: {p:+.3f} ({v:+.2f})" for n, (p, v) in joint_states.items())
    q = np.array([joint_states[n][0] for n in arm.names[:kinematics.N_JOINTS]])
    fk = kinematics.forward_kinematics(q)
    print(f"\r\033[K{line}", end='', flush=True)
    rpy = pin.rpy.matrixToRpy(fk.rotation)
    rot = np.array2string(fk.rotation.round(3).ravel(), max_line_width=1000)
    # print(f"\r\033[Kxyz = {fk.translation.round(4)} m   rpy = {rpy.round(3)} rad, rot = {rot}", end='', flush=True)


try:
    while True:
        camera.update()
        marker_pos = position_estimator.estimate(camera.read(), 1)
        # print(marker_pos)
        joint_states = arm.get_joint_states()
        log_state(joint_states)
        
        target = pin.SE3(np.array(TARGET_ROTATION), np.array(TARGET_POSITION))
        
        q_ik, ok = kinematics.inverse_kinematics(target)
        ik = np.append(q_ik, 0.0)  # j7 (gripper) = 0
        print(ik, ok)
        # if not ok:
        #     print("ik failed")
        #     break
        
        arm.set_joint_targets(ik)
        # arm.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
        
        
finally:
    arm.bus.disconnect()