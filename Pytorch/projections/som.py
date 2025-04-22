from minisom import MiniSom
import numpy as np
import matplotlib.pyplot as plt


def train_som(Z, x_dim=10, y_dim=10, sigma=1.0, learning_rate=0.5, iterations=1000, random_seed=None):
    """
    Train a MiniSom on the given latent vectors.

    :param Z: numpy array of shape (n_samples, n_features)
    :returns: trained MiniSom instance
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    som = MiniSom(x_dim, y_dim, Z.shape[1], sigma=sigma, learning_rate=learning_rate)
    som.random_weights_init(Z)
    som.train_random(Z, iterations)
    return som


def plot_u_matrix(som, figsize=(7, 7)):
    """
    Plot the U-Matrix (distance map) which shows the average distance
    between each neuron and its neighbors.
    """
    u_matrix = som.distance_map()
    plt.figure(figsize=figsize)
    plt.pcolor(u_matrix.T)  # transpose for correct orientation
    plt.colorbar(label='Distance')
    plt.title('U-Matrix (Distance Map)')
    plt.show()


def plot_data_hits(som, Z, labels=None, markers=None, colors=None, figsize=(7, 7)):
    """
    Plot where each data point falls on the map. Optionally color-code by label.

    :param labels: list of integers or categories for each sample
    :param markers: dict mapping label->marker
    :param colors: dict mapping label->color
    """
    # Determine map dimensions from weights
    weights = som.get_weights()
    x_dim, y_dim, _ = weights.shape

    plt.figure(figsize=figsize)
    for i, x in enumerate(Z):
        w = som.winner(x)
        if labels is not None and markers and colors:
            lbl = labels[i]
            plt.plot(
                w[0] + 0.5,
                w[1] + 0.5,
                markers.get(lbl, 'o'),
                markerfacecolor='None',
                markeredgecolor=colors.get(lbl, 'r'),
                markersize=12,
                markeredgewidth=2
            )
        else:
            plt.plot(
                w[0] + 0.5,
                w[1] + 0.5,
                'o', markerfacecolor='None', markeredgecolor='k'
            )
    plt.title('Data Hits on SOM')
    plt.xlim(0, x_dim)
    plt.ylim(0, y_dim)
    plt.show()


def plot_component_planes(som, figsize=(12, 12)):
    """
    Plot each feature's component plane in the SOM grid.
    """
    weights = som.get_weights()
    num_features = weights.shape[2]
    grid_x, grid_y, _ = weights.shape
    fig, axes = plt.subplots(
        int(np.ceil(np.sqrt(num_features))),
        int(np.ceil(np.sqrt(num_features))),
        figsize=figsize
    )
    axes = axes.flatten()

    for i in range(num_features):
        plane = weights[:, :, i]
        ax = axes[i]
        im = ax.pcolor(plane.T)
        ax.set_title(f'Feature {i}')
        fig.colorbar(im, ax=ax)

    # remove unused subplots
    for j in range(i + 1, len(axes)):
        fig.delaxes(axes[j])

    plt.tight_layout()
    plt.show()
