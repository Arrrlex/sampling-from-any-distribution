from sampling.protocols import Sampler
from sampling.linear_samplers import piecewise_linear_sampler
import jax
import numpy as np

def naive_envelope(domain, ymax=1):
    xmin, xmax = domain
    return piecewise_linear_sampler([(xmin, ymax), (xmax, ymax)])


# def piecewise_linear_envelope(f, domain, n):
#     xmin, xmax = domain
#     f_grad = jax.grad(f)
#     x_samples = np.linspace(xmin, xmax, n)  # Equally spaced samples in the domain
    
#     envelope_points = []  # List to store the envelope points

#     # Iterate through the sampled points
#     for i in range(n):
#         x_i = x_samples[i]
#         f_i = f(x_i)  # Evaluate the target function at x_i
#         grad_i = f_grad(x_i)  # Evaluate the gradient of the target function at x_i

#         # Calculate the tangent line extending from x_{i-1} to x_{i+1}
#         if i == 0:
#             x_prev = xmin
#             x_next = x_samples[i + 1]
#         elif i == n - 1:
#             x_prev = x_samples[i - 1]
#             x_next = xmax
#         else:
#             x_prev = x_samples[i - 1]
#             x_next = x_samples[i + 1]

#         # Calculate the equation of the tangent line
#         slope = grad_i
#         intercept = f_i - slope * x_i

#         # Calculate the intersection point of the tangent line with the domain
#         x_intersection = max(x_prev, min(x_next, -intercept / slope))

#         # Add the envelope point to the list
#         envelope_points.append((x_intersection.item(), f(x_intersection).item()))

#     return envelope_points


def piecewise_linear_envelope(f, domain, n, buffer=0.5):
    xmin, xmax = domain
    f_grad = jax.grad(f)
    x_samples = np.linspace(xmin, xmax, n)  # Equally spaced samples in the domain
    ys = f(x_samples)
    y_grads = [f_grad(x) for x in x_samples]
    
    envelope_points = []  # List to store the envelope points

    # Iterate through the sampled points, up to the second-to-last point
    for i in range(n - 1):
        x_i = x_samples[i]
        x_next = x_samples[i + 1]

        f_i = ys[i]
        f_next = ys[i+1]

        grad_i = y_grads[i]
        grad_next = y_grads[i+1]

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
        for i in range(n-1):
            x, y = envelope_points[i]
            y *= (1 + buffer)
            envelope_points[i] = (x, y)

    return envelope_points

# Example usage:
# Define your target function f and its gradient f_grad as numpy-compatible functions
# domain = (xmin, xmax)
# n = number of samples
# envelope_points = piecewise_linear_envelope(f, f_grad, domain, n)
