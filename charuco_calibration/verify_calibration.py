"""Live check of the calibration: shows the undistorted image and the board pose axes.

The axes should sit on the board corner and stay glued to it as you move the board.
Press q to quit.
"""
import cv2
import numpy as np

import config
from camera import open_camera
from load_calibration import load_calibration

K, dist = load_calibration()
board = config.make_board()
detector = cv2.aruco.CharucoDetector(board)
cap = open_camera()

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")

        view = cv2.undistort(frame, K, dist)  # corners are detected on the raw frame below
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, _, _ = detector.detectBoard(gray)

        if ids is not None and len(ids) >= config.MIN_CORNERS:
            obj, img = board.matchImagePoints(corners, ids)
            found, rvec, tvec = cv2.solvePnP(obj, img, K, dist)
            if found:
                # frame was undistorted, so project with zero distortion
                cv2.drawFrameAxes(view, K, np.zeros(5), rvec, tvec, config.SQUARE_LENGTH * 2)
                x, y, z = tvec.ravel()
                cv2.putText(view, f"board: x={x:+.3f} y={y:+.3f} z={z:+.3f} m", (10, 30),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("undistorted + pose (q to quit)", view)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
