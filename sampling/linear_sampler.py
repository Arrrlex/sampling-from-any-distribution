from dataclasses import dataclass
import random
from sampling.protocols import Distribution
from sampling.samplers import UniformSampler, RejectionSampler

@dataclass
class LinearSampler:
    """
    Sampler for a distribution that is linear between two points.
    """
    x_lower: float
    y_lower: float
    x_upper: float
    y_upper: float

    def draw(self) -> float:
