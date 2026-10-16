import math, time
import numpy as np
import pinocchio as pin
from robstride_dynamics import RobstrideBus, Motor, ParameterType
from constants import *
from ArmHardwareInterface import ArmHardware
from kinematics import KinematicsSolver

arm = ArmHardware("pcan0")
kinematics = KinematicsSolver("urdf/ReBot_Arm_RS.urdf")

arm.enable_all()

HOME_POSITION = [ 0.3006, -0.0201,  0.2197]
HOME_ROTATION = [[0.992,  0.121, -0.021], [-0.121,  0.992, -0.029],  [0.017,  0.032,  0.999]]

TARGET_POSITION = [0.1934, 0.1335, 0.3901]

# joint_states = arm.get_joint_states()
# q = np.array([joint_states[n][0] for n in arm.names[:kinematics.N_JOINTS]])
# fk = kinematics.forward_kinematics(q)


R = np.array([
    [0.651422,  0.741519, -0.160617],
    [0.727492, -0.550353,  0.409715],
    [0.215416, -0.383745, -0.897962],
])

target = pin.SE3(np.array(HOME_ROTATION), np.array(HOME_POSITION))


q_ik, ok = kinematics.inverse_kinematics(target)
ik = np.append(q_ik, 0.0)  # j7 (gripper) = 0
print(ik)
try:
    while True:
        joint_states = arm.get_joint_states()
        line = "  ".join(f"{n}: {p:+.3f} ({v:+.2f})" for n, (p, v) in joint_states.items())
        q = np.array([joint_states[n][0] for n in arm.names[:kinematics.N_JOINTS]])
        fk = kinematics.forward_kinematics(q)
        print(f"\r\033[K{line}", end='', flush=True)
        rpy = pin.rpy.matrixToRpy(fk.rotation)
        rot = np.array2string(fk.rotation.round(3).ravel(), max_line_width=1000)
        # print(f"\r\033[Kxyz = {fk.translation.round(4)} m   rpy = {rpy.round(3)} rad, rot = {rot}", end='', flush=True)
        arm.set_joint_targets(ik)
        # arm.set_joint_targets([0, 0, 0, 0, 0, 0, 0])
        
        
finally:
    arm.bus.disconnect()