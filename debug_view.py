"""Live debug window and run logger for the marker-tracking loop.
Keys: q / Esc quit, s save a snapshot, m mark an event in the log."""
import csv
import json
import os
import time
from collections import deque

import cv2
import numpy as np

FONT = cv2.FONT_HERSHEY_SIMPLEX
AA = cv2.LINE_AA

# BGR
BG = (22, 22, 22)
PANEL = (32, 32, 32)
GRID = (52, 52, 52)
TEXT = (225, 225, 225)
DIM = (135, 135, 135)
RED = (90, 90, 235)
GREEN = (120, 205, 120)
YELLOW = (70, 205, 235)
CYAN = (225, 200, 90)
MAGENTA = (210, 120, 220)
ORANGE = (60, 150, 245)
AXIS_COLORS = (RED, GREEN, (245, 160, 80))
AXIS_NAMES = "xyz"

W, H = 1280, 800
CAM_RECT = (0, 0, 640, 480)
STATUS_RECT = (0, 480, 640, 320)
TOP_RECT = (640, 0, 320, 320)
SIDE_RECT = (960, 0, 320, 320)
PLOT_RECT = (640, 320, 640, 480)

NAN3 = np.full(3, np.nan)

_xyz = lambda name: [f"{name}_{a}" for a in AXIS_NAMES]
_joints = lambda name: [f"{name}{i}" for i in range(1, 7)]
LOG_COLUMNS = (["t", "event", "frame", "marker_seen"] + _xyz("marker_cam") + _xyz("ee") + ["ee_roll", "ee_pitch", "ee_yaw"]
               + _xyz("target") + _xyz("err") + _xyz("step") + _xyz("marker_world") + ["ik_ok"]
               + _joints("q") + _joints("ik") + _joints("sent"))


def _pt(p):
    return tuple(int(round(float(np.clip(c, -10000, 10000)))) for c in p)


def _text(img, s, org, color=TEXT, scale=0.45, thickness=1):
    cv2.putText(img, s, _pt(org), FONT, scale, color, thickness, AA)


def _label(img, s, org, color=TEXT, scale=0.45, thickness=1):
    (w, h), base = cv2.getTextSize(s, FONT, scale, thickness)
    x, y = _pt(org)
    cv2.rectangle(img, (x - 3, y - h - 4), (x + w + 3, y + base + 2), (0, 0, 0), -1)
    cv2.putText(img, s, (x, y), FONT, scale, color, thickness, AA)


def _vec(v, fmt="{:+.3f}"):
    return "  ".join(f"{n} {fmt.format(x)}" for n, x in zip(AXIS_NAMES, v))


def _arrow(img, a, b, color, thickness=2):
    a, b = np.asarray(a, float), np.asarray(b, float)
    length = np.linalg.norm(b - a)
    if length < 2:
        return
    cv2.arrowedLine(img, _pt(a), _pt(b), color, thickness, AA, tipLength=min(0.4, 10 / length))


def _segments(mask):
    edges = np.flatnonzero(np.diff(np.concatenate([[0], mask.astype(np.int8), [0]])))
    return [slice(a, b) for a, b in zip(edges[::2], edges[1::2])]


class DebugView:
    def __init__(self, estimator, cam_to_ee, marker_ref, home_pos, arm,
                 marker_id=1, window="arm debug", history_s=10.0, view_half=0.25, snapshot_dir="debug_snapshots",
                 log_dir="logs", dead_zone=0.0, lost_timeout=None):
        self.estimator = estimator
        self.K = estimator.camera_matrix
        self.dist = estimator.dist_coeffs
        self.marker_length = 2 * float(estimator.object_points[1, 0])
        self.cam_to_ee = np.asarray(cam_to_ee, float)
        self.marker_ref = np.asarray(marker_ref, float)
        self.home = np.asarray(home_pos, float)
        self.dead_zone = dead_zone
        self.lost_timeout = lost_timeout
        self.arm = arm
        self.joint_limits = list(arm.limits.values())
        self.marker_id = marker_id
        self.window = window
        self.history_s = history_s
        self.view_half = view_half
        self.snapshot_dir = snapshot_dir

        self.history = deque()
        self.state = None
        self.last_t = None
        self.hz = 0.0
        self.last_seen = None
        self.ik_fails = 0
        self.window_open = False
        self.canvas = np.zeros((H, W, 3), np.uint8)

        self.start_t = None
        self.frame_index = 0
        self.events = 0
        self.pending_event = 0
        self.video = None
        self.run_dir = os.path.join(log_dir, time.strftime("%Y%m%d_%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self._write_config()
        self.log_file = open(os.path.join(self.run_dir, "log.csv"), "w", newline="", buffering=1)
        self.log = csv.writer(self.log_file)
        self.log.writerow(LOG_COLUMNS)
        print(f"logging to {self.run_dir}")

    def update(self, frame, marker_pos, q, fk, target_pos, q_ik, ik_ok):
        """Returns False when the user asked to quit."""
        self.record(frame, marker_pos, q, fk, target_pos, q_ik, ik_ok)
        self.render()
        cv2.imshow(self.window, self.canvas)
        self.window_open = True
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            self.save_snapshot()
        if key == ord("m"):
            self.events += 1
            self.pending_event = self.events
            cv2.imwrite(os.path.join(self.run_dir, f"event_{self.events}.png"), self.canvas)
            print(f"\nevent {self.events} marked")
        if key in (ord("q"), 27):
            return False
        return cv2.getWindowProperty(self.window, cv2.WND_PROP_VISIBLE) >= 1

    def record(self, frame, marker_pos, q, fk, target_pos, q_ik, ik_ok, now=None):
        now = time.monotonic() if now is None else now
        if self.last_t is not None and now > self.last_t:
            rate = 1.0 / (now - self.last_t)
            self.hz = rate if self.hz == 0 else 0.9 * self.hz + 0.1 * rate
        self.last_t = now

        ee, rot = fk.translation.copy(), fk.rotation.copy()
        seen = marker_pos is not None
        err = step = None
        marker_world = NAN3
        if seen:
            self.last_seen = now
            marker_pos = np.asarray(marker_pos, float)
            err = rot @ self.cam_to_ee @ (marker_pos - self.marker_ref)
            step = np.zeros(3) if self._in_dead_zone(marker_pos) else err
            # ignores the camera's offset from the EE origin, constant while orientation is held
            marker_world = ee + rot @ self.cam_to_ee @ marker_pos
        if not ik_ok:
            self.ik_fails += 1

        self.history.append((now, ee, np.array(target_pos, float), NAN3 if err is None else err,
                             marker_world, seen, bool(ik_ok)))
        while now - self.history[0][0] > self.history_s:
            self.history.popleft()

        self.state = dict(frame=frame, marker_pos=marker_pos, q=np.asarray(q, float), ee=ee,
                          target=np.array(target_pos, float), err=err, step=step,
                          q_ik=np.asarray(q_ik, float), ik_ok=bool(ik_ok), marker_world=marker_world, now=now)
        self._log_row(frame, rot)

    def render(self):
        self.canvas[:] = BG
        h = self._history_arrays()
        self._draw_camera(self._sub(CAM_RECT))
        self._draw_view(self._sub(TOP_RECT), h, axes=(1, 0), signs=(-1, -1), title="TOP  (looking down)",
                        legend=[("EE", TEXT, "ring"), ("target", CYAN, "dot"), ("marker", MAGENTA, "square")])
        self._draw_view(self._sub(SIDE_RECT), h, axes=(0, 2), signs=(1, -1), title="SIDE  (looking from -y)",
                        legend=[("error", YELLOW, "arrow"), ("correction", GREEN, "arrow")])
        self._draw_plots(self._sub(PLOT_RECT), h)
        self._draw_status(self._sub(STATUS_RECT), h)
        for x, y, w, hh in (CAM_RECT, STATUS_RECT, TOP_RECT, SIDE_RECT, PLOT_RECT):
            cv2.rectangle(self.canvas, (x, y), (x + w - 1, y + hh - 1), GRID, 1)
        return self.canvas

    def save_snapshot(self):
        os.makedirs(self.snapshot_dir, exist_ok=True)
        path = os.path.join(self.snapshot_dir, time.strftime("arm_debug_%Y%m%d_%H%M%S.png"))
        cv2.imwrite(path, self.canvas)
        print(f"\nsaved {path}")

    def close(self):
        if self.window_open:
            cv2.destroyWindow(self.window)
            self.window_open = False
        if self.video is not None:
            self.video.release()
            self.video = None
        if not self.log_file.closed:
            self.log_file.close()
            print(f"\nlog saved to {self.run_dir}")

    def _write_config(self):
        arm = self.arm
        config = dict(
            cam_to_ee=self.cam_to_ee.tolist(), marker_ref=self.marker_ref.tolist(), home_pos=self.home.tolist(),
            marker_id=self.marker_id, marker_length=self.marker_length,
            camera_matrix=np.asarray(self.K).tolist(), dist_coeffs=np.asarray(self.dist).ravel().tolist(),
            joint_limits=getattr(arm, "limits", None),
            arm={k: getattr(arm, k, None) for k in ("KP", "KD", "TORQUE_LIMIT", "MAX_VEL", "MAX_ERROR", "MAX_DT")},
        )
        with open(os.path.join(self.run_dir, "config.json"), "w") as f:
            json.dump(config, f, indent=2)

    def _log_row(self, frame, rot):
        s = self.state
        if self.start_t is None:
            self.start_t = s["now"]
        frame_id = ""
        if frame is not None:
            if self.video is None:
                fh, fw = frame.shape[:2]
                self.video = cv2.VideoWriter(os.path.join(self.run_dir, "camera.avi"),
                                             cv2.VideoWriter_fourcc(*"MJPG"), 30, (fw, fh))
            self.video.write(frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR))
            frame_id = self.frame_index
            self.frame_index += 1

        # last command sent to the motors after the speed limiter (set_joint_targets runs after update)
        names = self.arm.names[:len(s["q"])]
        sent = [getattr(self.arm._limiters[n], "commanded", None) for n in names]
        rpy = (np.arctan2(rot[2, 1], rot[2, 2]), -np.arcsin(np.clip(rot[2, 0], -1, 1)), np.arctan2(rot[1, 0], rot[0, 0]))

        def f(v):
            return [] if v is None else ["" if x is None or not np.isfinite(x) else f"{x:.5f}" for x in v]

        none3 = [None] * 3
        self.log.writerow(
            [f"{s['now'] - self.start_t:.4f}", self.pending_event or "", frame_id, int(s["marker_pos"] is not None)]
            + f(none3 if s["marker_pos"] is None else s["marker_pos"]) + f(s["ee"]) + f(rpy) + f(s["target"])
            + f(none3 if s["err"] is None else s["err"]) + f(none3 if s["step"] is None else s["step"])
            + f(s["marker_world"]) + [int(s["ik_ok"])] + f(s["q"]) + f(s["q_ik"]) + f(sent))
        self.pending_event = 0

    def _sub(self, rect):
        x, y, w, h = rect
        return self.canvas[y:y + h, x:x + w]

    def _history_arrays(self):
        cols = list(zip(*self.history))
        return dict(t=np.array(cols[0]) - self.state["now"], ee=np.array(cols[1]), target=np.array(cols[2]),
                    err=np.array(cols[3]), marker_world=np.array(cols[4]),
                    seen=np.array(cols[5], bool), ik_ok=np.array(cols[6], bool))

    def _in_dead_zone(self, marker_pos):
        return np.linalg.norm(marker_pos - self.marker_ref) <= self.dead_zone

    def _lost_text(self):
        lost_for = self.state["now"] - self.last_seen
        going_home = self.lost_timeout is not None and lost_for > self.lost_timeout
        return lost_for, "going home" if going_home else "target frozen"

    def _detect(self, frame):
        """Corners and rotation for drawing; PositionEstimator only returns the position."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        corners, ids, _ = self.estimator.detector.detectMarkers(gray)
        if ids is None:
            return None
        for marker_corners, found_id in zip(corners, ids.ravel()):
            if found_id == self.marker_id:
                ok, rvec, tvec = cv2.solvePnP(self.estimator.object_points, marker_corners.reshape(4, 2),
                                              self.K, self.dist, flags=cv2.SOLVEPNP_IPPE_SQUARE)
                return (marker_corners, rvec, tvec) if ok else None
        return None

    def _project(self, p, K):
        px, _ = cv2.projectPoints(np.asarray(p, np.float64).reshape(-1, 3), np.zeros(3), np.zeros(3), K, self.dist)
        return px.reshape(-1, 2)

    def _mapping_text(self):
        parts = []
        for j, n in enumerate(AXIS_NAMES):
            col = self.cam_to_ee[:, j]
            i = int(np.argmax(np.abs(col)))
            parts.append(f"cam +{n} -> EE {'+' if col[i] > 0 else '-'}{AXIS_NAMES[i]}")
        return "   ".join(parts)

    def _sign_check(self, h):
        """Slope of error vs. EE position per axis. With a still marker and a correct CAM_TO_EE it is -1."""
        seen = h["seen"]
        if seen.sum() < 10:
            return None
        ee, err = h["ee"][seen], h["err"][seen]
        out = []
        for i in range(3):
            x = ee[:, i] - ee[:, i].mean()
            y = err[:, i] - err[:, i].mean()
            if np.sqrt((x ** 2).mean()) < 0.004:
                out.append((None, "move the arm along this axis", DIM))
                continue
            slope = (x * y).sum() / (x * x).sum()
            if abs(slope + 1) < 0.35:
                out.append((slope, "OK", GREEN))
            elif abs(slope) < 0.3:
                out.append((slope, "no response: wrong axis in CAM_TO_EE?", YELLOW))
            elif slope > 0:
                out.append((slope, "FLIPPED: sign wrong in CAM_TO_EE", RED))
            else:
                out.append((slope, f"scale {-slope:.1f}x: check marker_length", YELLOW))
        return out

    def _draw_camera(self, panel):
        s = self.state
        ph, pw = panel.shape[:2]
        frame = s["frame"]
        if frame is None:
            panel[:] = PANEL
            _text(panel, "NO CAMERA FRAME", (20, 40), RED, 0.8, 2)
            return

        img = frame if frame.ndim == 3 else cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        fh, fw = img.shape[:2]
        sx, sy = pw / fw, ph / fh
        img = cv2.convertScaleAbs(cv2.resize(img, (pw, ph)), alpha=0.75)
        K = np.diag([sx, sy, 1.0]) @ self.K

        cv2.drawMarker(img, _pt((K[0, 2], K[1, 2])), DIM, cv2.MARKER_CROSS, 24, 1, AA)

        marker = s["marker_pos"]
        holding = marker is not None and self._in_dead_zone(marker)
        ref_ok = self.marker_ref[2] > 1e-3
        ref_px = None
        if ref_ok:
            ref_px = self._project(self.marker_ref, K)[0]
            self._draw_ref(img, K, ref_px, GREEN if holding else CYAN)

        det = self._detect(frame) if marker is not None else None
        if det is not None:
            corners, rvec, tvec = det
            cv2.aruco.drawDetectedMarkers(img, [(corners.reshape(1, 4, 2) * [sx, sy]).astype(np.float32)], None, GREEN)
            cv2.drawFrameAxes(img, K, self.dist, rvec, tvec, self.marker_length * 0.5, 2)
        if marker is not None:
            center = self._project(marker, K)[0]
            cv2.circle(img, _pt(center), 5, GREEN, -1, AA)
            if ref_px is not None and not holding:
                _arrow(img, center, ref_px, YELLOW, 2)

        if marker is not None:
            hold = "  IN DEAD ZONE (holding)" if self._in_dead_zone(marker) else ""
            _label(img, f"MARKER  dist {np.linalg.norm(marker):.3f} m{hold}", (10, 22), GREEN, 0.55)
        elif self.last_seen is None:
            _label(img, "MARKER NEVER SEEN", (10, 22), RED, 0.55)
        else:
            lost_for, action = self._lost_text()
            _label(img, f"MARKER LOST  {lost_for:.1f} s  ({action})", (10, 22), RED, 0.55)
        if not ref_ok:
            _label(img, f"MARKER_REF z={self.marker_ref[2]:+.2f}: camera z is depth, ref must be [x, y, dist>0]",
                   (10, 46), RED, 0.42)

        y0 = ph - 64
        _label(img, "camera frame (x right, y down, z fwd)", (10, y0), DIM, 0.4)
        _label(img, "seen " + (_vec(marker) if marker is not None else "-"), (10, y0 + 18), GREEN, 0.42)
        _label(img, "ref  " + _vec(self.marker_ref), (10, y0 + 36), CYAN, 0.42)
        if marker is not None:
            _label(img, "diff " + _vec(marker - self.marker_ref), (10, y0 + 54), YELLOW, 0.42)
        _label(img, self._mapping_text(), (pw - 360, ph - 10), DIM, 0.38)

        self._draw_depth_gauge(img, marker)
        panel[:] = img

    def _draw_ref(self, img, K, ref_px, color):
        ph, pw = img.shape[:2]
        c = np.asarray(ref_px, float)

        if self.dead_zone > 0:
            radius = max(int(K[0, 0] * self.dead_zone / self.marker_ref[2]), 3)
            shade = img.copy()
            cv2.circle(shade, _pt(c), radius, color, -1, AA)
            cv2.addWeighted(shade, 0.22, img, 0.78, 0, dst=img)
            cv2.circle(img, _pt(c), radius, color, 2, AA)
            _label(img, f"dead zone {self.dead_zone * 100:.1f} cm", c + (radius * 0.72 + 4, -radius * 0.72 - 4), color, 0.42)

        # expected marker outline at MARKER_REF, as corner brackets
        hl = self.marker_length / 2
        square = self._project(self.marker_ref + np.array([[-hl, -hl, 0], [hl, -hl, 0], [hl, hl, 0], [-hl, hl, 0]]), K)
        if np.ptp(square[:, 0]) < 1.5 * pw and np.ptp(square[:, 1]) < 1.5 * ph:
            for i in range(4):
                p, prev, nxt = square[i], square[i - 1], square[(i + 1) % 4]
                cv2.line(img, _pt(p), _pt(p + (prev - p) * 0.25), color, 2, AA)
                cv2.line(img, _pt(p), _pt(p + (nxt - p) * 0.25), color, 2, AA)
        else:
            _label(img, "expected marker outline bigger than the view: check marker_length", (10, 46), YELLOW, 0.42)

        for d in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            d = np.array(d, float)
            cv2.line(img, _pt(c + d * 7), _pt(c + d * 24), (0, 0, 0), 4, AA)
            cv2.line(img, _pt(c + d * 7), _pt(c + d * 24), color, 2, AA)
        cv2.circle(img, _pt(c), 3, color, -1, AA)
        _label(img, f"MARKER_REF  {self.marker_ref[2]:.2f} m", c + (-62, 44), color, 0.45)

    def _draw_depth_gauge(self, img, marker):
        ph, pw = img.shape[:2]
        x, top, bottom = pw - 26, 50, ph - 90
        depth = None if marker is None else marker[2]
        far = max(0.2, self.marker_ref[2], depth or 0) * 1.4
        to_y = lambda d: bottom - (np.clip(d, 0, far) / far) * (bottom - top)
        cv2.rectangle(img, (x - 6, top), (x + 6, bottom), (0, 0, 0), -1)
        cv2.rectangle(img, (x - 6, top), (x + 6, bottom), DIM, 1)
        _label(img, "depth", (x - 22, top - 10), DIM, 0.38)
        if self.marker_ref[2] > 0:
            color = GREEN if marker is not None and self._in_dead_zone(marker) else CYAN
            if self.dead_zone > 0:
                y0, y1 = to_y(self.marker_ref[2] + self.dead_zone), to_y(self.marker_ref[2] - self.dead_zone)
                cv2.rectangle(img, _pt((x - 6, y0)), _pt((x + 6, max(y1, y0 + 2))), color, -1)
            y = to_y(self.marker_ref[2])
            cv2.line(img, _pt((x - 12, y)), _pt((x + 12, y)), color, 2)
        if depth is not None:
            y = to_y(depth)
            cv2.circle(img, _pt((x, y)), 6, GREEN, -1, AA)
            d = depth - self.marker_ref[2]
            _label(img, f"{'too far' if d > 0 else 'too close'} {abs(d):.3f}", (x - 120, y + 4), YELLOW, 0.4)

    def _draw_view(self, panel, h, axes, signs, title, legend):
        s = self.state
        ph, pw = panel.shape[:2]
        panel[:] = PANEL
        ha, va = axes
        hs, vs = signs
        cx, cy = pw / 2, ph / 2 - 4
        scale = (min(pw, ph) / 2 - 30) / self.view_half

        def px(p):
            return (cx + hs * (p[ha] - self.home[ha]) * scale, cy + vs * (p[va] - self.home[va]) * scale)

        for k in np.arange(-self.view_half, self.view_half + 1e-9, 0.05):
            color = (70, 70, 70) if abs(k) < 1e-9 else GRID
            cv2.line(panel, _pt((cx + k * scale, 0)), _pt((cx + k * scale, ph)), color, 1)
            cv2.line(panel, _pt((0, cy + k * scale)), _pt((pw, cy + k * scale)), color, 1)

        cv2.drawMarker(panel, _pt(px(self.home)), DIM, cv2.MARKER_TILTED_CROSS, 10, 1, AA)
        _text(panel, "home", np.add(px(self.home), (6, 14)), DIM, 0.35)

        for key, color in (("ee", (150, 150, 150)), ("target", (150, 130, 60))):
            pts = np.array([px(p) for p in h[key]], np.int32).reshape(-1, 1, 2)
            if len(pts) > 1:
                cv2.polylines(panel, [pts], False, color, 1, AA)
        for p in h["marker_world"][h["seen"]][::3]:
            cv2.circle(panel, _pt(px(p)), 1, (120, 70, 130), -1)

        if s["marker_pos"] is not None:
            m = np.array(px(s["marker_world"]))
            edge = np.clip(m, 12, (pw - 12, ph - 12))
            if np.allclose(m, edge):
                cv2.rectangle(panel, _pt(m - 5), _pt(m + 5), MAGENTA, -1)
            else:
                inward = (np.array([cx, cy]) - edge) / max(np.linalg.norm(np.array([cx, cy]) - edge), 1e-9)
                _arrow(panel, edge + inward * 14, edge, MAGENTA, 2)
                dist = np.linalg.norm(s["marker_world"] - s["ee"])
                _text(panel, f"marker {dist:.2f} m", np.clip(edge + inward * 34 - (36, -4), 4, (pw - 90, ph - 30)),
                      MAGENTA, 0.38)

        if s["err"] is not None:
            _arrow(panel, px(s["ee"]), px(s["ee"] + s["err"]), YELLOW, 2)
            _arrow(panel, px(s["target"] - s["step"]), px(s["target"]), GREEN, 2)
        cv2.circle(panel, _pt(px(s["target"])), 5, CYAN, -1, AA)
        cv2.circle(panel, _pt(px(s["ee"])), 8, TEXT, 2, AA)

        _text(panel, title, (8, 16), TEXT, 0.42)
        _text(panel, "grid 5 cm", (pw - 66, 16), DIM, 0.35)

        o = np.array([22.0, ph - 46])
        for axis, direction in ((ha, (hs, 0)), (va, (0, vs))):
            tip = o + np.array(direction) * 20
            _arrow(panel, o, tip, AXIS_COLORS[axis], 2)
            _text(panel, f"+{AXIS_NAMES[axis]}", tip + np.array(direction) * 6 + (-6, 4), AXIS_COLORS[axis], 0.38)

        x = 8
        for name, color, kind in legend:
            y = ph - 10
            if kind == "ring":
                cv2.circle(panel, (x + 5, y - 4), 5, color, 2, AA)
            elif kind == "dot":
                cv2.circle(panel, (x + 5, y - 4), 4, color, -1, AA)
            elif kind == "square":
                cv2.rectangle(panel, (x + 1, y - 8), (x + 9, y), color, -1)
            elif kind == "arrow":
                cv2.arrowedLine(panel, (x, y - 4), (x + 14, y - 4), color, 2, AA, tipLength=0.4)
            elif kind == "box":
                cv2.rectangle(panel, (x, y - 9), (x + 12, y + 1), color, 1)
            _text(panel, name, (x + 18, y), DIM, 0.38)
            x += 26 + cv2.getTextSize(name, FONT, 0.38, 1)[0][0]

    def _draw_plots(self, panel, h):
        ph, pw = panel.shape[:2]
        panel[:] = PANEL
        strip = ph // 3
        lag = np.linalg.norm(h["target"] - h["ee"], axis=1)[:, None]
        self._plot(panel[0:strip], h["t"], h["err"], "error, base frame (m)")
        self._plot(panel[strip:2 * strip], h["t"], h["target"] - self.home, "target - home (m)")
        self._plot(panel[2 * strip:], h["t"], lag, "|target - EE|  tracking lag (m)", colors=(TEXT,),
                   symmetric=False, bands=(("marker", h["seen"]), ("IK", h["ik_ok"])), time_axis=True)

    def _plot(self, img, t, Y, title, colors=AXIS_COLORS, hlines=(), symmetric=True, bands=(), time_axis=False):
        ph, pw = img.shape[:2]
        left, right, top = 52, 110, 20
        bottom = ph - 8 - 12 * len(bands) - (14 if time_axis else 0)
        w = pw - left - right

        finite = Y[np.isfinite(Y)]
        m = max([np.abs(finite).max() if finite.size else 0.0] + [abs(v) for v in hlines])
        m = max(m * 1.15, 0.005)
        lo = -m if symmetric else 0.0
        to_u = lambda tt: left + (tt + self.history_s) / self.history_s * w
        to_v = lambda y: bottom - (y - lo) / (m - lo) * (bottom - top)

        cv2.rectangle(img, (left, top), (left + w, bottom), GRID, 1)
        if symmetric:
            cv2.line(img, _pt((left, to_v(0))), _pt((left + w, to_v(0))), (70, 70, 70), 1)
        for v in hlines:
            for u in range(left, left + w, 8):
                cv2.line(img, _pt((u, to_v(v))), _pt((u + 4, to_v(v))), ORANGE, 1)
        _text(img, title, (left, top - 6), TEXT, 0.42)
        _text(img, f"{m:+.3f}", (4, top + 8), DIM, 0.35)
        _text(img, f"{lo:+.3f}", (4, bottom), DIM, 0.35)

        for k in range(Y.shape[1]):
            color = colors[k % len(colors)]
            for seg in _segments(np.isfinite(Y[:, k])):
                pts = np.stack([to_u(t[seg]), to_v(Y[seg, k])], axis=1)
                if len(pts) > 1:
                    cv2.polylines(img, [np.clip(pts, -10000, 10000).astype(np.int32).reshape(-1, 1, 2)], False, color, 1, AA)
            last = Y[-1, k] if len(Y) else np.nan
            name = AXIS_NAMES[k] if Y.shape[1] > 1 else "now"
            _text(img, f"{name} " + ("-" if not np.isfinite(last) else f"{last:+.4f}"),
                  (left + w + 10, top + 12 + 16 * k), color, 0.42)

        for i, (name, mask) in enumerate(bands):
            y = bottom + 6 + 12 * i
            _text(img, name, (4, y + 8), DIM, 0.33)
            for seg in _segments(mask):
                cv2.rectangle(img, _pt((to_u(t[seg.start]), y)), _pt((to_u(t[seg.stop - 1]) + 1, y + 8)), GREEN, -1)
            for seg in _segments(~mask):
                cv2.rectangle(img, _pt((to_u(t[seg.start]), y)), _pt((to_u(t[seg.stop - 1]) + 1, y + 8)), RED, -1)

        if time_axis:
            for sec in np.arange(-self.history_s, 0.01, 2.0):
                _text(img, f"{sec:.0f}s" if sec < 0 else "now", (to_u(sec) - 10, ph - 6), DIM, 0.33)

    def _draw_status(self, panel, h):
        s = self.state
        panel[:] = PANEL
        x, y, dy = 12, 22, 19

        def line(text, color=TEXT, scale=0.45):
            nonlocal y
            _text(panel, text, (x, y), color, scale)
            y += dy

        line(f"loop {self.hz:4.1f} Hz    marker seen {100 * h['seen'].mean():3.0f}% of last {self.history_s:.0f}s    "
             f"IK fails {self.ik_fails}", GREEN if s["ik_ok"] else RED)
        line("base frame (m)", DIM, 0.38)
        line("EE      " + _vec(s["ee"]))
        line("target  " + _vec(s["target"]), CYAN)
        offset = s["target"] - self.home
        line("offset  " + _vec(offset))
        if s["err"] is not None:
            line("error   " + _vec(s["err"]) + f"   |e| {np.linalg.norm(s['err']) * 1000:.0f} mm", YELLOW)
            hold = "   dead zone: holding" if self._in_dead_zone(s["marker_pos"]) else ""
            line("step    " + _vec(s["step"]) + hold, GREEN)
        else:
            action = self._lost_text()[1] if self.last_seen is not None else "target frozen"
            line(f"error   -   (marker not seen, {action})", DIM)
            line("step    -", DIM)
        lag = np.linalg.norm(s["target"] - s["ee"])
        line(f"lag     |target - EE| {lag * 1000:.0f} mm")

        y += 4
        line("CAM_TO_EE check (keep the marker still, move the arm):", DIM, 0.38)
        checks = self._sign_check(h)
        flipped = []
        if checks is None:
            line("  not enough marker samples yet", DIM, 0.4)
        else:
            for i, (slope, verdict, color) in enumerate(checks):
                slope_s = "  -  " if slope is None else f"{slope:+.2f}"
                line(f"  {AXIS_NAMES[i]}  slope {slope_s}  (want -1)  {verdict}", color, 0.4)
                if color == RED:
                    flipped.append(AXIS_NAMES[i])
        mw = h["marker_world"][h["seen"]]
        if len(mw) > 5:
            spread = np.sqrt(np.mean(np.sum((mw - mw.mean(axis=0)) ** 2, axis=1))) * 1000
            line(f"  marker world spread {spread:.0f} mm  (small = consistent)", GREEN if spread < 15 else YELLOW, 0.4)

        hints = []
        if self.marker_ref[2] <= 0:
            hints.append("MARKER_REF z <= 0: ref must be in front of the camera")
        if flipped:
            hints.append(f"axis {' '.join(flipped)} looks flipped in CAM_TO_EE")
        if not s["ik_ok"]:
            hints.append("IK failed this loop: command not sent")
        for i, hint in enumerate(hints[:2]):
            _text(panel, "! " + hint, (x, panel.shape[0] - 30 + 16 * i), YELLOW, 0.4)

        self._draw_joints(panel, 400)

    def _draw_joints(self, panel, x0):
        s = self.state
        pw = panel.shape[1]
        _text(panel, "joints (rad)", (x0, 22), DIM, 0.38)
        _text(panel, "measured", (x0 + 90, 22), TEXT, 0.38)
        _text(panel, "IK", (x0 + 160, 22), CYAN, 0.38)
        x1 = pw - 14
        for i in range(len(s["q"])):
            lo, hi = self.joint_limits[i]
            y = 44 + i * 40
            q, cmd = s["q"][i], s["q_ik"][i]
            to_x = lambda a: x0 + (np.clip(a, lo, hi) - lo) / (hi - lo) * (x1 - x0)
            near = min(q - lo, hi - q) < 0.05 * (hi - lo)
            _text(panel, f"j{i + 1}  {q:+.3f} -> {cmd:+.3f}", (x0, y), RED if near else TEXT, 0.4)
            cv2.rectangle(panel, (x0, y + 6), (x1, y + 16), (55, 55, 55), -1)
            cv2.rectangle(panel, (x0, y + 6), (x1, y + 16), RED if near else GRID, 1)
            cv2.line(panel, _pt((to_x(cmd), y + 2)), _pt((to_x(cmd), y + 20)), CYAN, 2)
            cv2.line(panel, _pt((to_x(q), y + 4)), _pt((to_x(q), y + 18)), TEXT, 2)
