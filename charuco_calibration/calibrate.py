"""Run the ChArUco calibration on the images in images/ and save the result to camera_calibration.npz."""
import sys

import cv2
import numpy as np

import config

paths = sorted(config.IMAGES_DIR.glob("*.png"))
if not paths:
    sys.exit(f"no images in {config.IMAGES_DIR}, run capture_images.py first")

board = config.make_board()
detector = cv2.aruco.CharucoDetector(board)

obj_points, img_points, used, image_size = [], [], [], None
for path in paths:
    image = cv2.imread(str(path))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    if image_size is None:
        image_size = gray.shape[::-1]
    elif gray.shape[::-1] != image_size:
        print(f"skip {path.name}: size {gray.shape[::-1]} differs from {image_size}")
        continue

    corners, ids, _, _ = detector.detectBoard(gray)
    if ids is None or len(ids) < config.MIN_CORNERS:
        print(f"skip {path.name}: {0 if ids is None else len(ids)} corners")
        continue

    obj, img = board.matchImagePoints(corners, ids)
    obj_points.append(obj)
    img_points.append(img)
    used.append(path.name)

print(f"using {len(used)} of {len(paths)} images")
if len(used) < 10:
    sys.exit("need at least 10 usable images, capture more")

rms, K, dist, rvecs, tvecs = cv2.calibrateCamera(obj_points, img_points, image_size, None, None)

# per-image reprojection error, to spot bad images
print("\nper-image RMS reprojection error (px):")
errors = []
for name, obj, img, rvec, tvec in zip(used, obj_points, img_points, rvecs, tvecs):
    proj, _ = cv2.projectPoints(obj, rvec, tvec, K, dist)
    err = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img.reshape(-1, 2)) ** 2, axis=1))))
    errors.append(err)
    print(f"  {name}: {err:.3f}")

print(f"\nimage size: {image_size[0]}x{image_size[1]}")
print(f"overall RMS reprojection error: {rms:.4f} px  (below ~0.5 is good)")
print(f"camera matrix:\n{K}")
print(f"distortion coefficients (k1 k2 p1 p2 k3):\n{dist.ravel()}")

np.savez(
    config.OUTPUT_FILE,
    camera_matrix=K,
    dist_coeffs=dist,
    image_size=np.array(image_size),
    rms=rms,
)
print(f"\nsaved {config.OUTPUT_FILE}")
