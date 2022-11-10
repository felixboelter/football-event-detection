import torch
import torch.nn as nn


class CrossEntropyLoss(nn.Module):
    def __init__(self, config=None):
        super().__init__()
        self.cel = nn.CrossEntropyLoss(reduction='none')
        self.cfg = config
        self.start_int = (self.cfg.window_len - self.cfg.prediction_len) // 2

    def forward(self, yhat, y):
        yhat = yhat[:, :, self.start_int: self.start_int + y.shape[1]]
        # y = y[:, self.start_int: self.start_int + self.cfg.prediction_len]

        loss = torch.mean(
            self.cel(yhat, y).mean(1))
        return loss

