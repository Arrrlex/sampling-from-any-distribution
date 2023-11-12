from typing_extensions import Protocol

class Distribution(Protocol):
    def pdf(self, x: float) -> float:
        ...

class Sampler(Protocol):
    total_area: float
    domain: tuple[float, float]

    def f(self, x: float) -> float:
        ...

    def draw(self) -> float:
        ...
