import numpy as np

from moljourney.space import run_pca


def test_run_pca():
    # Test we can run the PCA method.
    X = np.random.normal(loc=10, scale=5, size=(10, 10))
    _, scores, scaler = run_pca(X, scale=False, n_components=None)
    assert scaler is None
    assert scores.shape == (10, 10)
    _, scores, scaler = run_pca(X, scale=False, n_components=11)
    assert scores.shape == (10, 10)
    _, scores, scaler = run_pca(X, scale=True, n_components=2)
    assert scaler is not None
    assert scores.shape == (10, 2)
