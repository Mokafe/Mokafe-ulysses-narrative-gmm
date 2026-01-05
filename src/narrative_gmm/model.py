from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from sklearn.mixture import GaussianMixture

@dataclass
class SemiSupGMMResult:
    gmm: GaussianMixture
    resp: np.ndarray

def fit_semisup_gmm(
    X: np.ndarray,
    labels: np.ndarray | None,
    n_components: int = 2,
    seed: int = 0,
    alpha: float = 20.0,
    cov_type: str = "full",
    max_iter: int = 300,
) -> SemiSupGMMResult:
    gmm = GaussianMixture(
        n_components=n_components,
        covariance_type=cov_type,
        random_state=seed,
        max_iter=max_iter,
        n_init=3,
        reg_covar=1e-6,
    )
    gmm.fit(X)
    resp = gmm.predict_proba(X)

    if labels is not None:
        y = labels.copy()
        mask = y >= 0
        if mask.any():
            K = resp.shape[1]
            boost = np.ones_like(resp)
            for k in range(K):
                boost[mask & (y == k), k] = np.exp(alpha)
            resp = resp * boost
            resp = resp / resp.sum(axis=1, keepdims=True)
    return SemiSupGMMResult(gmm=gmm, resp=resp)
