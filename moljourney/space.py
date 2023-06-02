"""Methods for dimension reduction."""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def run_pca(
    data: np.ndarray,
    scale: bool = True,
    n_components: int | float | None = None,
) -> tuple[type[PCA], np.ndarray, type[StandardScaler] | None]:
    """Run a principal component analysis.

    Parameters
    ----------
    data : object like numpy.array
        The raw data to run PCA on.
    scale : boolean, optional
        If True, a standard scaler will be applied to the data.
    n_components : int or float, optional
        Select the number of components to use.
    """
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(data)
    else:
        scaler = None
        X = data
    if n_components is not None and n_components > X.shape[1]:
        n_components = X.shape[1]
    pca = PCA(n_components=n_components)
    scores = pca.fit_transform(X)
    return pca, scores, scaler
