from dataclasses import dataclass
import random
from sampling.protocols import Distribution
from sampling.samplers import UniformSampler, RejectionSampler

@dataclass
class SimpleZigguratSampler:
    """Ziggurat Sampler, assuming the pdf starts at 0 and is decreasing."""
    dist: Distribution
    upper: float = 1000
    buckets: list[int]

    @classmethod
    def from_dist(cls, dist: Distribution, n: int = 1000):
        """Create a Ziggurat Sampler from a distribution."""
        buckets = cls._create_buckets(dist, n)
        return cls(dist, buckets=buckets)

    @staticmethod
    def _create_buckets(dist: Distribution, n: int) -> list[int]:
        """Create the buckets for a Ziggurat Sampler."""
        ...

    def draw(self) -> float:
        """Draw a sample from the distribution."""
        bucket = random.randint(0, len(self.buckets) - 1)
        if bucket == 0:
            return self._draw_from_tail()

        return self._draw_from_bucket(bucket)

    def _draw_from_tail(self) -> float:
        """Draw a sample from the tail of the distribution."""
        ...

    def _draw_from_bucket(self, bucket: int) -> float:
        """Draw a sample from the given bucket."""
        while True:
            y_sampler = UniformSampler(self.buckets[bucket - 1], self.buckets[bucket])

            x, y = self.x_sampler.draw(), self.y_sampler.draw()
            if self.dist.pdf(x) > y:
                return x