import cv2
import numpy as np


class PositionEstimator:
    def __init__(self, marker_length, calibration_file="charuco_calibration/camera_calibration.npz",
                 dictionary=cv2.aruco.DICT_5X5_100):
        data = np.load(calibration_file)
        self.camera_matrix = data["camera_matrix"]
        self.dist_coeffs = data["dist_coeffs"]

        self.detector = cv2.aruco.ArucoDetector(cv2.aruco.getPredefinedDictionary(dictionary))

        # marker corners in the marker frame (origin at the center), in detector order: TL, TR, BR, BL
        h = marker_length / 2
        self.object_points = np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], dtype=np.float32)

    def estimate(self, frame, marker_id):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.detector.detectMarkers(gray)
        if ids is None:
            return None

        for marker_corners, found_id in zip(corners, ids.ravel()):
            if found_id != marker_id:
                continue
            ok, _, tvec = cv2.solvePnP(
                self.object_points, marker_corners.reshape(4, 2),
                self.camera_matrix, self.dist_coeffs, flags=cv2.SOLVEPNP_IPPE_SQUARE,
            )
            return tvec.ravel() if ok else None
        return None
