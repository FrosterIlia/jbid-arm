import math, time
from robstride_dynamics import RobstrideBus, Motor, ParameterType
from constants import *

NAME = "j1"
bus = RobstrideBus("pcan0", motors={NAME: Motor(id=1, model="rs-05")})
bus.connect()



def send(target, kp, kd, vel=0.0):
    bus.write_operation_frame(NAME, position=target, kp=kp, kd=kd, velocity=vel, torque=0)
    return bus.read_operation_frame(NAME)        # (pos, vel, torque, temp)

start = bus.read(NAME, ParameterType.MECHANICAL_POSITION)
print(f"start = {start:+.3f} rad")

bus.disable(NAME)
bus.set_run_mode(NAME, 0)    # MIT mode
bus.enable(NAME)
bus.write(NAME, ParameterType.TORQUE_LIMIT, 3.0)   # Nm
try:
    # 1) Soft start: hold where it already is, ramping stiffness 0 -> KP over 1 s
    for i in range(101):
        send(start, KP * i / 100, KD)
        time.sleep(DT)

    # 2) Smooth move: out 0.2 rad and back over T seconds
    T, A = 15.0, 0.1
    w = 2 * math.pi / T
    t0 = time.perf_counter()
    while (t := time.perf_counter() - t0) < T:
        # target = start + A * (1 - math.cos(w * t))
        # vel = A * w * math.sin(w * t)            # derivative of target
        pos, v, tau, temp = send(0, KP, KD, 0)
        err =  0
        print(f"t={t:4.2f}  target={0:+.3f}  pos={pos:+.3f}  err={err:+.3f}  tau={tau:+.2f}")
        if abs(err) > 0.15:
            raise RuntimeError("tracking error too large: blocked or too weak")
        time.sleep(DT)
finally:
    bus.disable(NAME)
    bus.disconnect(disable_torque=False)