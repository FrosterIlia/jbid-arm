import time


def clamp(x, lo, hi):
    return min(max(x, lo), hi)


class SetpointLimiter:
    def __init__(self, lower, upper, max_vel, max_error, max_dt=0.1):
        self.lower = lower
        self.upper = upper
        self.max_vel = max_vel
        self.max_error = max_error
        self.max_dt = max_dt
        self.commanded = None
        self._last_time = None

    def reset(self, position):
        self.commanded = position
        self._last_time = time.monotonic()

    def update(self, goal, measured):
        if self.commanded is None:
            self.reset(measured)

        now = time.monotonic()
        max_step = self.max_vel * min(now - self._last_time, self.max_dt)
        self._last_time = now

        goal = clamp(goal, self.lower, self.upper)
        cmd = self.commanded + clamp(goal - self.commanded, -max_step, max_step)
        cmd = clamp(cmd, measured - self.max_error, measured + self.max_error)
        self.commanded = clamp(cmd, self.lower, self.upper)
        return self.commanded
