import cv2


class CameraStream:
    def __init__(self, index=0, width=640, height=480, fps=30, fourcc="YUYV"):
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"could not open camera {index} (check /dev/video*)")

        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

        self.frame = None

    def update(self):
        ok, frame = self.cap.read()
        if ok:
            self.frame = frame
        return ok

    def read(self):
        return self.frame

    def release(self):
        self.cap.release()
