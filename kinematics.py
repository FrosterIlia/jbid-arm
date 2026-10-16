import numpy as np
import pinocchio as pin


class KinematicsSolver:
    N_JOINTS = 6

    def __init__(self, urdf_path, ee_frame="gripper_end"):
        full_model = pin.buildModelFromUrdf(urdf_path)

        joint_names = list(full_model.names[1:])
        if len(joint_names) < self.N_JOINTS:
            raise ValueError(f"URDF has {len(joint_names)} joints, expected at least {self.N_JOINTS}")
        self.joint_names = joint_names[: self.N_JOINTS]

        locked = [full_model.getJointId(name) for name in joint_names[self.N_JOINTS :]]
        self.model = pin.buildReducedModel(full_model, locked, pin.neutral(full_model))
        self.data = self.model.createData()

        if not self.model.existFrame(ee_frame):
            raise ValueError(f"frame {ee_frame!r} not found in {urdf_path}")
        self.ee_frame_id = self.model.getFrameId(ee_frame)

        self.lower_limit = self.model.lowerPositionLimit.copy()
        self.upper_limit = self.model.upperPositionLimit.copy()

    def _check_q(self, q):
        q = np.asarray(q, dtype=float)
        if q.shape != (self.N_JOINTS,):
            raise ValueError(f"expected {self.N_JOINTS} joint angles, got shape {q.shape}")
        return q

    def forward_kinematics(self, q):
        q = self._check_q(q)
        pin.framesForwardKinematics(self.model, self.data, q)
        return self.data.oMf[self.ee_frame_id].copy()

    def inverse_kinematics(self, target, q0=None, max_iters=1000, tol=1e-4, dt=0.1, damping=1e-6):
        q = np.clip(self._check_q(q0) if q0 is not None else pin.neutral(self.model), self.lower_limit, self.upper_limit)

        for _ in range(max_iters):
            pin.framesForwardKinematics(self.model, self.data, q)
            iMd = self.data.oMf[self.ee_frame_id].actInv(target)
            err = pin.log(iMd).vector
            if np.linalg.norm(err) < tol:
                return q, True

            J = pin.computeFrameJacobian(self.model, self.data, q, self.ee_frame_id)
            J = -pin.Jlog6(iMd.inverse()) @ J
            v = -J.T @ np.linalg.solve(J @ J.T + damping * np.eye(6), err)
            q = np.clip(pin.integrate(self.model, q, v * dt), self.lower_limit, self.upper_limit)

        return q, False
