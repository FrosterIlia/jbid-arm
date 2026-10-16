"""Helper for other scripts: K, dist = load_calibration()."""
import numpy as np

import config


def load_calibration(path=None):
    data = np.load(path or config.OUTPUT_FILE)
    return data["camera_matrix"], data["dist_coeffs"]
