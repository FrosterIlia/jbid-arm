"""Render the board from config.py to a PNG, in case you need to print another one."""
import cv2

import config

PIXELS_PER_SQUARE = 200
MARGIN = 40

board = config.make_board()
size = (config.SQUARES_X * PIXELS_PER_SQUARE, config.SQUARES_Y * PIXELS_PER_SQUARE)
image = board.generateImage(size, marginSize=MARGIN)
path = config.HERE / "charuco_board.png"
cv2.imwrite(str(path), image)
print(f"saved {path}")
print(f"print it so that one square is exactly {config.SQUARE_LENGTH * 1000:.1f} mm "
      f"(then measure it and update SQUARE_LENGTH / MARKER_LENGTH in config.py)")
