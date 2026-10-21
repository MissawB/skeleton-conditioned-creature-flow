import copy
import torch
from torch import nn

class EMA:
    """
    Exponential moving average of model weights, used for sampling instead of the last
    optimizer state (which can land right after a bad update).
    With warmup, the decay is min(decay, (1 + n) / (10 + n)) so early averages are not
    dominated by the random initialization.
    """

    def __init__(self, model: nn.Module, decay: float = 0.999, warmup: bool = True):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)
        self.decay = decay
        self.warmup = warmup
        self.num_updates = 0

    @torch.no_grad()
    def update(self, model: nn.Module):
        decay = min(self.decay, (1 + self.num_updates) / (10 + self.num_updates)) if self.warmup else self.decay
        self.num_updates += 1
        for ema_p, p in zip(self.model.parameters(), model.parameters()):
            ema_p.lerp_(p.detach(), 1.0 - decay)
        for ema_b, b in zip(self.model.buffers(), model.buffers()):
            ema_b.copy_(b)

    def state_dict(self):
        return self.model.state_dict()
