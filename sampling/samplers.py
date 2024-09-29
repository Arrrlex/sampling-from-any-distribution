from dataclasses import dataclass
from functools import cached_property
import random
from typing import Callable

from sampling.protocols import Distribution


@dataclass
class UniformSampler:
    lower: float
    upper: float

    def draw(self) -> float:
        width = self.upper - self.lower
        return self.lower + random.random() * width


@dataclass
class FunctionSampler:
    func: Callable[[], float]

    def draw(self) -> float:
        return self.func()


@dataclass
class RejectionSampler:
    dist: Distribution
    lower: float = -1000
    upper: float = 1000

    @cached_property
    def x_sampler(self):
        return UniformSampler(self.lower, self.upper)

    @cached_property
    def y_sampler(self):
        return UniformSampler(0, 1)

    def draw(self) -> float:
        while True:
            x, y = self.x_sampler.draw(), self.y_sampler.draw()
            if self.dist.pdf(x) > y:
                return x