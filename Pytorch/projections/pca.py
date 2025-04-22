# Pytorch/projections/pca.py
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


def fit_pca(Z, n_components, additional_params=None):
    """
    Fit PCA on the given latent vectors.

    :param Z: array-like, shape (n_samples, n_features)
    :param n_components: target number of principal components
    :param additional_params: dict of extra PCA parameters (e.g. svd_solver)
    :returns: fitted PCA instance and transformed data of shape (n_samples, n_components)
    """
    params = additional_params or {}
    pca = PCA(n_components=n_components, **params)
    X_reduced = pca.fit_transform(Z)
    return pca, X_reduced


def plot_explained_variance(pca, figsize=(6,4)):
    """
    Plot cumulative explained variance ratio vs number of components.
    """
    cum_var = np.cumsum(pca.explained_variance_ratio_)
    plt.figure(figsize=figsize)
    plt.plot(range(1, len(cum_var)+1), cum_var, marker='o')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.title('PCA Explained Variance')
    plt.grid(True)
    plt.show()


def plot_principal_components(pca, feature_names=None, figsize=(8,6)):
    """
    Bar plots for each principal component's loadings.
    """
    comps = pca.components_
    n_comp, n_feat = comps.shape
    fig, axes = plt.subplots(int(np.ceil(np.sqrt(n_comp))),
                             int(np.ceil(np.sqrt(n_comp))),
                             figsize=figsize)
    axes = axes.flatten()
    for i in range(n_comp):
        ax = axes[i]
        comps_i = comps[i]
        idx = np.arange(n_feat)
        ax.bar(idx, comps_i)
        ax.set_title(f'PC {i+1}')
        if feature_names is not None:
            ax.set_xticks(idx)
            ax.set_xticklabels(feature_names, rotation=90)
    # remove unused axes
    for j in range(i+1, len(axes)):
        fig.delaxes(axes[j])
    plt.tight_layout()
    plt.show()