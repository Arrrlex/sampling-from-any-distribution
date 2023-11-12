from sampling.protocols import Sampler
import numpy as np
import random
import math
import matplotlib.pyplot as plt
from scipy.stats import norm
import imageio
import os
from tqdm import tqdm
from dataclasses import dataclass
from typing import Callable


class TriangleSampler(Sampler):
    def __init__(self, x1, y1, x2, y2):
        assert y1 == 0 or y2 == 0, "One of y1 or y2 must be 0"
        self.domain = (x1, x2)
        slope = (y2 - y1) / (x2 - x1)
        self.base = x2 - x1
        self.height = max(y1, y2)
        self.total_area = 0.5 * self.base * self.height
        self.x1, self.y1, self.x2, self.y2 = x1, y1, x2, y2
        self.slope = slope

    def f(self, x):
        if self.x1 <= x <= self.x2:
            return self.slope * (x - self.x1) + self.y1
        else:
            return 0

    def draw(self):
        r = random.uniform(0, self.total_area)
        if self.y1 == 0:
            return self.x1 + math.sqrt(2 * r / abs(self.slope))
        else:
            return self.x2 - math.sqrt(2 * (self.total_area - r) / abs(self.slope))


class TranslatedSampler(Sampler):
    def __init__(self, sampler, h):
        self.sampler = sampler
        self.h = h
        self.domain = sampler.domain
        # Total area is the sum of the original area and the area of the added rectangle.
        self.added_area = (self.domain[1] - self.domain[0]) * h
        self.total_area = sampler.total_area + self.added_area

    def f(self, x):
        return self.sampler.f(x) + self.h

    def draw(self):
        # Decide whether to sample from the original distribution or the added rectangle.
        if random.uniform(0, self.total_area) < self.added_area:
            # Sample uniformly from the domain.
            return random.uniform(self.domain[0], self.domain[1])
        else:
            # Sample from the original distribution.
            return self.sampler.draw()


def create_segment_sampler(p1, p2):
    """Create a sampler for a segment between points p1 and p2."""
    x1, y1 = p1
    x2, y2 = p2

    base_height = min(y1, y2)
    triangle_sampler = TriangleSampler(x1, y1 - base_height, x2, y2 - base_height)
    if base_height > 0:
        return TranslatedSampler(triangle_sampler, base_height)
    else:
        return triangle_sampler


class ComposedSampler(Sampler):
    def __init__(self, samplers):
        # Check if the domains of the samplers align properly
        for i in range(len(samplers) - 1):
            assert samplers[i].domain[1] == samplers[i + 1].domain[0], "Domains of consecutive samplers must align."

        self.samplers = samplers
        # Calculate the total area of all samplers combined
        self.total_area = sum(sampler.total_area for sampler in samplers)
        # The domain of the composed sampler
        self.domain = (samplers[0].domain[0], samplers[-1].domain[1])

    def f(self, x):
        # Find the appropriate sampler for the given x and use its f function
        for sampler in self.samplers:
            if sampler.domain[0] <= x <= sampler.domain[1]:
                return sampler.f(x)
        return 0

    def draw(self):
        # Choose a sampler based on their relative areas
        r = random.uniform(0, self.total_area)
        cumulative_area = 0
        for sampler in self.samplers:
            cumulative_area += sampler.total_area
            if r <= cumulative_area:
                return sampler.draw()

def piecewise_linear_sampler(points):
    samplers = []
    for i in range(len(points) - 1):
        samplers.append(create_segment_sampler(points[i], points[i+1]))
    return ComposedSampler(samplers)