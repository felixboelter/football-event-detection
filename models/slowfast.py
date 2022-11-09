import torch
import torch.nn as nn
from pytorchvideo.models.hub import slowfast_r50
import random
random.seed(0)


class CustomSlowFast(nn.Module):
    def __init__(self, model, config):
        super(CustomSlowFast, self).__init__()
        self.model = model
        self.cfg = config
        self.blocks = model.blocks[:5]

        self.pool = nn.MaxPool3d(kernel_size=(1, 8 // self.cfg.size_fact, 8 // self.cfg.size_fact))
        self.drop1 = nn.Dropout(0.0)
        self.lstm = nn.LSTM(2304, 128, num_layers=1, 
                        batch_first=True, bidirectional=True)
        self.drop2 = nn.Dropout(0.0)
        self.pred = nn.Linear(2304, 4)

    def forward(self, x):
        for _, block in enumerate(self.blocks):
            x = block(x)

        x_slow, x_fast = x[0], x[1]
        x_slow = x_slow.repeat_interleave(self.cfg.slow_fast_alpha, dim=2)

        x = torch.cat([x_slow, x_fast], dim=1)

        x = self.pool(x).view(x.shape[0], x.shape[1], -1).permute(0, 2, 1)
        x = self.drop1(x)
        # x = self.lstm(x)[0]
        # x = self.drop2(x)
        x = self.pred(x).permute(0, 2, 1)

        return x


def create_model(config):
    base_model = slowfast_r50(False)
    model = CustomSlowFast(base_model, config)
    model.to(config.device)

    return model
