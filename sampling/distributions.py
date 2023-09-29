from dataclasses import dataclass
from functools import cached_property
import numpy as np


@dataclass
class Triangle:
    """
    A probability distribution where probability density increases
    linearly from `l` to `u`, and is 0 everywhere else.
    """
    l: float
    u: float

    def pdf(self, x):
        if not self.l < x < self.u:
            return 0
        return 2 * (x - self.l) / (self.u - self.l) ** 2

@dataclass
class Normal:
    mean: float
    std: float

    @cached_property
    def beta(self):
        return self.std ** -2

    def pdf(self, x):
        exponent = -2 * self.beta * (x - self.mean) ** 2
        multiplier = self.beta / np.sqrt(2 * np.pi)
        return multiplier * np.exp(exponent)