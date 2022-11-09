import numpy as np


class AccMetric:
    def __init__(self, config=None):
        self.cfg = config
        self.start_int = (self.cfg.window_len - self.cfg.prediction_len) // 2

    def get_error(self, yhat, y):
        yhat = yhat[:, :, self.start_int: self.start_int + y.shape[1]].argmax(1)
        # y = y[:, self.start_int: self.start_int + self.cfg.prediction_len]

        score = np.mean((y == yhat).sum(1) / self.cfg.prediction_len)

        return score
