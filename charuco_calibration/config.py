from pathlib import Path

import cv2

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------- board
# TODO: fill in with the parameters of your physical board.
SQUARES_X = 5            # number of squares along the horizontal direction
SQUARES_Y = 7            # number of squares along the vertical direction
SQUARE_LENGTH = 0.034    # side of one chessboard square, in meters
MARKER_LENGTH = 0.026    # side of one ArUco marker, in meters
ARUCO_DICT = cv2.aruco.DICT_5X5_100
LEGACY_PATTERN = False  # True if the board was made by an old OpenCV (< 4.6) / online generator

# --------------------------------------------------------------- camera
CAMERA_INDEX = 0         # /dev/video<N>
FRAME_WIDTH = 640
FRAME_HEIGHT = 480
FPS = 30
FOURCC = "YUYV"          # "YUYV" (raw, clean) or "MJPG" (compressed, gave corrupt-JPEG warnings here)

# ---------------------------------------------------------------- paths
IMAGES_DIR = HERE / "images"
OUTPUT_FILE = HERE / "camera_calibration.npz"

# Minimum number of detected ChArUco corners for an image to be used
MIN_CORNERS = 12


def make_board():
    board = cv2.aruco.CharucoBoard(
        (SQUARES_X, SQUARES_Y),
        SQUARE_LENGTH,
        MARKER_LENGTH,
        cv2.aruco.getPredefinedDictionary(ARUCO_DICT),
    )
    board.setLegacyPattern(LEGACY_PATTERN)
    return board


def make_detector():
    return cv2.aruco.CharucoDetector(make_board())
