from collections import deque


class SmoothedBest:
    """Track argmax of the 5-point moving average of a score, keeping its state.

    Falls back to plain argmax when fewer than 5 epochs are recorded.
    """

    WINDOW = 5
    HALF = WINDOW // 2

    def __init__(self, window=WINDOW):
        self.window = window
        self.half = window // 2
        self._ring = deque(maxlen=window)
        self.scores = []
        self.best_smoothed = None
        self.best_index = None
        self.best_state = None
        self.best_raw = None
        self.best_raw_index = None
        self.best_raw_state = None

    def update(self, epoch, score, state_fn):
        """Record `score` for `epoch`. `state_fn()` is called at most once."""
        self.scores.append(float(score))
        state = state_fn()
        if self.best_raw is None or score > self.best_raw:
            self.best_raw = float(score)
            self.best_raw_index = epoch
            self.best_raw_state = state

        self._ring.append((epoch, float(score), state))
        if len(self._ring) < self.window:
            return
        smoothed = sum(s for _, s, _ in self._ring) / self.window
        centre_epoch, _, centre_state = self._ring[self.half]
        # Strictly-greater keeps the first maximum.
        if self.best_smoothed is None or smoothed > self.best_smoothed:
            self.best_smoothed = smoothed
            self.best_index = centre_epoch
            self.best_state = centre_state

    def resolve(self):
        """(selected_epoch, state_dict). Uses the raw argmax for runs under 5 epochs."""
        if self.best_state is not None:
            return self.best_index, self.best_state
        return self.best_raw_index, self.best_raw_state
