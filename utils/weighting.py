import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl


def clipper(x, a_min, a_max):
    return np.clip(x, a_min=a_min, a_max=a_max, out=x)


def linear(x, offset, step_pos, k=1):
    weights = np.ones_like(x)
    for i in range(x.shape[0]):
        for j in range(x.shape[1]):
            x_ = x[i, j]
            if x_ < step_pos:
                weights[i, j] = 1 + k * (x_ - step_pos)
            else:
                weights[i, j] = 1

    return weights


def central_axis(y, k=0.01):
    return 1 - np.tanh(k * y**2)


def smooth_sink(x, y, x0, y0, k=0.1):
    return 1 - 1 / (k * ((x - x0) ** 2 + (y - y0) ** 2) + 1)


def get_k_parameters(x, y):
    x_range = np.max(x) - np.min(x)
    y_range = np.max(y) - np.min(y)
    k_x = 1 / x_range
    k_y = 0.5 / ((y_range / 2) ** 2)
    k_circ = 100 / (x_range**2 + y_range**2)
    return k_x, k_y, k_circ


def combined_weighting(x_grid, y_grid, x_min, step_pos, clip_min=0.3):
    k_x, k_y, k_circ = get_k_parameters(x_grid, y_grid)
    weights_linear_x = linear(x_grid, x_min, step_pos=step_pos, k=k_x)
    weights_central_axis_y = central_axis(y_grid, k_y)
    weights_smooth_sink = smooth_sink(x_grid, y_grid, 0, 0, k=k_circ)

    weights = weights_linear_x * weights_central_axis_y * weights_smooth_sink
    weights = clipper(weights, a_min=clip_min, a_max=1)
    weights = weights.flatten()
    return weights


if __name__ == "__main__":
    x = np.linspace(-30, 100, 100)
    y = np.linspace(-30, 30, 100)

    x_min = np.min(x)
    x_max = np.max(x)
    y_min = np.min(y)
    y_max = np.max(y)

    xx, yy = np.meshgrid(x, y, indexing="ij")

    from scipy.optimize import minimize

    y_range = np.max(y) - np.min(y)
    k_y_2 = 0.5 / ((y_range / 2) ** 2)
    k_x, k_y, k_circ = get_k_parameters(x, y)

    print(k_x, k_y, k_circ)
    print(k_y_2 - k_y)  # check to see if the better math gives the same result

    weights_linear_x = linear(xx, x_min, step_pos=20, k=k_x)
    weights_central_axis_y = central_axis(yy, k_y)
    weights_smooth_sink = smooth_sink(xx, yy, 0, 0, k=k_circ)

    weights_combined = weights_linear_x * weights_central_axis_y * weights_smooth_sink
    weights_combined_clipped = clipper(weights_combined, 0.3, 1)

    print(weights_combined_clipped.min(), weights_combined_clipped.max())

    fig, axes = plt.subplots(2, 2, figsize=(11, 5))
    axes = axes.flatten()
    for ax, weights, title in zip(
        axes,
        [
            weights_linear_x,
            weights_central_axis_y,
            weights_smooth_sink,
            weights_combined_clipped,
        ],
        ["linear_x", "central_axis_y", "smooth_sink", "combined"],
    ):
        # ax.imshow(weights, origin='lower')
        ax.contourf(xx, yy, weights)
        ax.colorbar = plt.colorbar(ax.contourf(xx, yy, weights), ax=ax)
        ax.axis("equal")
        ax.set_title(title)

    plt.tight_layout()
    plt.figure()
    plt.imshow(weights_combined_clipped.T, origin="lower")
    plt.axis("equal")
    plt.colorbar()

    # %% Sampling

    flattened_weights = weights_combined_clipped.flatten()
    xx_flattened = xx.flatten()
    yy_flattened = yy.flatten()
    idxs = np.arange(len(xx_flattened))

    # sample points with weights
    n_samples = 1000000
    sample_indices = np.random.choice(
        idxs, n_samples, p=flattened_weights / np.sum(flattened_weights)
    )

    fig = plt.figure(figsize=(8, 3))
    plt.hist2d(xx_flattened[sample_indices], yy_flattened[sample_indices], bins=100)
    plt.colorbar()
    plt.axis("equal")
    plt.tight_layout()
    plt.show()
