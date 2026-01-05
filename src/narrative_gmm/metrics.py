from __future__ import annotations
import numpy as np
import pandas as pd

def entropy_from_resp(resp: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    p = np.clip(resp, eps, 1.0)
    return -np.sum(p * np.log(p), axis=1)

def margin_from_resp(resp: np.ndarray) -> np.ndarray:
    s = np.sort(resp, axis=1)
    return s[:, -1] - s[:, -2] if resp.shape[1] >= 2 else np.zeros(resp.shape[0], dtype=float)

def turning_points(resp: np.ndarray, top_k: int = 20) -> np.ndarray:
    H = entropy_from_resp(resp)
    M = margin_from_resp(resp)
    Hn = (H - H.min()) / (H.max() - H.min() + 1e-12)
    Mn = (M - M.min()) / (M.max() - M.min() + 1e-12)
    score = Hn + (1.0 - Mn)
    idx = np.argsort(-score)[:top_k]
    return np.sort(idx)

def attach_uncertainty(df: pd.DataFrame, resp: np.ndarray) -> pd.DataFrame:
    out = df.copy()
    out["cluster"] = resp.argmax(axis=1)
    out["p_max"] = resp.max(axis=1)
    out["entropy"] = entropy_from_resp(resp)
    out["margin"] = margin_from_resp(resp)
    return out
