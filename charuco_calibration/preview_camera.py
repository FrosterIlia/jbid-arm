"""Live camera view. Use it to check focus, exposure and the camera index. Press q to quit."""
import cv2

from camera import open_camera

cap = open_camera()
try:
    while True:
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("camera returned no frame")
        cv2.imshow("preview (q to quit)", frame)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
finally:
    cap.release()
    cv2.destroyAllWindows()
