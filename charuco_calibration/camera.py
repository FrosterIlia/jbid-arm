import cv2

import config


def open_camera(index=None, width=None, height=None, fps=None):
    cap = cv2.VideoCapture(config.CAMERA_INDEX if index is None else index, cv2.CAP_V4L2)
    if not cap.isOpened():
        raise RuntimeError(f"could not open camera {config.CAMERA_INDEX} (check /dev/video*)")

    # MJPG allows higher resolutions / frame rates over USB than the raw YUYV default
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*config.FOURCC))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width or config.FRAME_WIDTH)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height or config.FRAME_HEIGHT)
    cap.set(cv2.CAP_PROP_FPS, fps or config.FPS)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != ((width or config.FRAME_WIDTH), (height or config.FRAME_HEIGHT)):
        print(f"WARNING: camera is running at {w}x{h}, not the requested resolution")
    return cap
