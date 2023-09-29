from typing_extensions import Protocol

class Distribution(Protocol):
    def pdf(self, x: float) -> float:
        ...

class Sampler(Protocol):
    def draw(self) -> float:
        ...