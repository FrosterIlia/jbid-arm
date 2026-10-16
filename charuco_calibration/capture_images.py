"""Capture calibration images.

SPACE  save the current frame (only when enough board corners are detected)
q      quit

Move the board around: fill the whole frame, get close and far, near the edges and corners,
and tilt it in all directions. Aim for 20-40 varied images. Keep the camera focus fixed.
"""
import time

import cv2

import config
from camera import open_camera

config.IMAGES_DIR.mkdir(exist_ok=True)
detector = config.make_detector()
cap = open_camera()
count = len(list(config.IMAGES_DIR.glob("*.png")))

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        corners, ids, marker_corners, marker_ids = detector.detectBoard(gray)
        n = 0 if ids is None else len(ids)
        good = n >= config.MIN_CORNERS

        view = frame.copy()
        if marker_ids is not None:
            cv2.aruco.drawDetectedMarkers(view, marker_corners, marker_ids)
        if ids is not None:
            # OpenCV 5 returns corners as (N, 2); the draw function wants (N, 1, 2)
            cv2.aruco.drawDetectedCornersCharuco(view, corners.reshape(-1, 1, 2), ids, (0, 255, 0))
        color = (0, 255, 0) if good else (0, 0, 255)
        cv2.putText(view, f"corners: {n}  saved: {count}  [SPACE] save  [q] quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
        cv2.imshow("capture", view)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        if key == ord(" "):
            if not good:
                print(f"not saved: only {n} corners detected (need {config.MIN_CORNERS})")
                continue
            path = config.IMAGES_DIR / f"img_{time.strftime('%Y%m%d_%H%M%S')}_{count:03d}.png"
            cv2.imwrite(str(path), frame)  # save the clean frame, without overlays
            count += 1
            print(f"saved {path.name} ({n} corners)")
finally:
    cap.release()
    cv2.destroyAllWindows()
