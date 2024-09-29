from sampling.protocols import Sampler
from sampling.linear_samplers import (
    piecewise_linear_sampler,
    max_piecewise_linear_function,
)
import jax
import jax.numpy as jnp
from typing import Callable


def naive_envelope(domain, ymax=1) -> Sampler:
    xmin, xmax = domain
    return piecewise_linear_sampler([(xmin, ymax), (xmax, ymax)])


def piecewise_linear_envelope_using_tangents(
    f: Callable,
    domain: tuple[float, float],
    n: int,
    buffer: float = 0.0,
) -> Sampler:
    xmin, xmax = domain
    f_grad = jax.grad(f)
    x_samples = jnp.linspace(xmin, xmax, n)  # Equally spaced samples in the domain
    ys = f(x_samples)
    y_grads = [f_grad(x) for x in x_samples]

    envelope_points = []  # List to store the envelope points

    # Iterate through the sampled points, up to the second-to-last point
    for i in range(n - 1):
        x_i = x_samples[i]
        x_next = x_samples[i + 1]

        f_i = ys[i]
        f_next = ys[i + 1]

        grad_i = y_grads[i]
        grad_next = y_grads[i + 1]

        # Calculate the slope and intercept of the tangent line at x_i
        slope_i = grad_i
        intercept_i = f_i - slope_i * x_i

        # Calculate the slope and intercept of the tangent line at x_{i+1}
        slope_next = grad_next
        intercept_next = f_next - slope_next * x_next

        # Calculate the intersection point of the tangent lines
        x_intersection = (intercept_next - intercept_i) / (slope_i - slope_next)
        y_intersection = slope_i * x_intersection + intercept_i

        # Add the intersection point to the list
        envelope_points.append((x_intersection.item(), y_intersection.item()))

    if buffer > 0:
        for i in range(n - 1):
            x, y = envelope_points[i]
            y *= 1 + buffer
            envelope_points[i] = (x, y)

    return piecewise_linear_sampler(envelope_points)


def piecewise_linear_envelope_naive(
    f: Callable,
    domain: tuple[float, float],
    n: int,
) -> Sampler:
    """
    Create a piecewise linear estimate for f by choosing n points in the domain and joining them up.
    """
    xmin, xmax = domain
    x_samples = jnp.linspace(xmin, xmax, n)
    y_samples = f(x_samples)

    return piecewise_linear_sampler(list(zip(x_samples, y_samples)))


def piecewise_linear_envelope_max(
    f: Callable,
    domain: tuple[float, float],
    n: int,
) -> Sampler:
    """
    Create a piecewise linear envelope for f by choosing n points in the domain and joining them up.
    """
    envelope1 = piecewise_linear_envelope_using_tangents(f, domain, n)
    envelope2 = piecewise_linear_envelope_naive(f, domain, n)
    return max_piecewise_linear_function(envelope1, envelope2)
