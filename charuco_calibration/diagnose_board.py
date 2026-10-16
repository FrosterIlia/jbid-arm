"""Figure out why corners are not detected although markers are.

Hold the whole board still in front of the camera and run this. It grabs a few frames, finds the ArUco
dictionary that matches your markers, then tries board sizes and the legacy flag and reports the
settings that give the most ChArUco corners. Put the winning values into config.py.
"""
import cv2

import config
from camera import open_camera

DICTS = sorted(n for n in dir(cv2.aruco) if n.startswith("DICT_"))

cap = open_camera()
gray = None
best_markers = 0
for _ in range(40):
    ok, frame = cap.read()
    if not ok:
        continue
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    found = cv2.aruco.ArucoDetector(
        cv2.aruco.getPredefinedDictionary(config.ARUCO_DICT)
    ).detectMarkers(g)[1]
    n = 0 if found is None else len(found)
    if n >= best_markers:
        best_markers, gray = n, g
cap.release()

if gray is None:
    raise SystemExit("no frames from the camera")

print(f"configured dictionary {config.ARUCO_DICT}: {best_markers} markers seen in the best frame")

# 1. which dictionary matches the printed markers?
print("\ndictionary search (markers detected):")
results = []
for name in DICTS:
    d = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))
    _, ids, _ = cv2.aruco.ArucoDetector(d).detectMarkers(gray)
    results.append((0 if ids is None else len(ids), name, None if ids is None else sorted(ids.ravel().tolist())))
results.sort(reverse=True)
for n, name, ids in results[:5]:
    print(f"  {name}: {n}" + (f"  ids={ids}" if n else ""))
best_dict_name = results[0][1]
if results[0][0] == 0:
    raise SystemExit("no markers detected with any dictionary: board not in view or image out of focus")

# 2. which board layout gives ChArUco corners?
dictionary = cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, best_dict_name))
layouts = []
for sx in range(3, 13):
    for sy in range(3, 13):
        for legacy in (False, True):
            board = cv2.aruco.CharucoBoard((sx, sy), 0.03, 0.022, dictionary)
            board.setLegacyPattern(legacy)
            corners, ids, _, _ = cv2.aruco.CharucoDetector(board).detectBoard(gray)
            n = 0 if ids is None else len(ids)
            if n:
                layouts.append((n, sx, sy, legacy))
layouts.sort(reverse=True)

print(f"\nbest dictionary: {best_dict_name}")
print("board layouts by ChArUco corners found:")
if not layouts:
    print("  none: no layout produced corners (marker ids may not match a standard ChArUco layout)")
for n, sx, sy, legacy in layouts[:8]:
    print(f"  SQUARES_X={sx} SQUARES_Y={sy} LEGACY_PATTERN={legacy}: {n} corners")
