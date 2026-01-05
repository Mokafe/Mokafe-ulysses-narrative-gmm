from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .metrics import attach_uncertainty, turning_points
from .model import fit_semisup_gmm
from .io import load_events_json, load_events_csv

@dataclass
class ReportConfig:
    n_components: int = 2
    seed: int = 0
    alpha: float = 20.0
    top_k: int = 20
    context_w: int = 3
    plots: bool = True

def _save_fig(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()

def build_report(input_path: str | Path, out_dir: str | Path, cfg: ReportConfig) -> dict[str, Path]:
    """
    input_path: .json (Ulysses_fixed.json) or .csv (ulysses_stream.csv)
    """
    in_path = Path(input_path)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)

    if in_path.suffix.lower() == ".csv":
        loaded = load_events_csv(in_path)
    else:
        loaded = load_events_json(in_path)

    df = loaded.df_events.copy()
    X = loaded.X

    # labels: accept label column if present; otherwise all -1
    labels = np.full(len(df), -1, dtype=int)
    if "label" in df.columns:
        for i, v in enumerate(df["label"].to_list()):
            if pd.isna(v) or v is None:
                continue
            try:
                labels[i] = int(v)
            except Exception:
                pass

    res = fit_semisup_gmm(X, labels=labels, n_components=cfg.n_components, seed=cfg.seed, alpha=cfg.alpha)
    df_pred = attach_uncertainty(df, res.resp)

    # Save CSVs
    p_events = out / "events_all.csv"
    p_preds = out / "preds_all.csv"
    df.to_csv(p_events, index=False, encoding="utf-8")
    df_pred.to_csv(p_preds, index=False, encoding="utf-8")

    # Turning points (TopK)
    idx_tp = turning_points(res.resp, top_k=cfg.top_k)
    df_tp = df_pred.loc[idx_tp].copy()
    df_tp["tp_rank"] = range(1, len(df_tp) + 1)
    p_tp = out / f"boundary_top{cfg.top_k}.csv"
    df_tp.to_csv(p_tp, index=False, encoding="utf-8")

    # Context window (+/- w)
    ctx_rows = []
    for r, i in enumerate(idx_tp, start=1):
        lo = max(0, i - cfg.context_w)
        hi = min(len(df_pred) - 1, i + cfg.context_w)
        tmp = df_pred.loc[lo:hi].copy()
        tmp["tp_rank"] = r
        if "global_step" in df_pred.columns:
            tmp["tp_center_step"] = int(df_pred.loc[i, "global_step"])
        ctx_rows.append(tmp)
    df_ctx = pd.concat(ctx_rows, axis=0).reset_index(drop=True) if ctx_rows else pd.DataFrame()
    p_ctx = out / "boundary_context.csv"
    df_ctx.to_csv(p_ctx, index=False, encoding="utf-8")

    # Chapter/episode summaries if available
    group_key = None
    for k in ("chapter", "episode", "doc_id"):
        if k in df_pred.columns:
            group_key = k
            break

    p_density = out / "scene_turning_density.csv"
    p_trans = out / "scene_cluster_transitions.csv"
    if group_key is not None:
        g = df_pred.groupby(group_key, dropna=False)
        df_density = g["entropy"].sum().rename("entropy_sum").to_frame()
        df_density["n_events"] = g.size()
        df_density["entropy_mean"] = g["entropy"].mean()
        df_density["turning_density"] = df_density["entropy_sum"] / (df_density["n_events"] + 1e-12)
        df_density.reset_index().to_csv(p_density, index=False, encoding="utf-8")

        df_pred["_cluster_prev"] = df_pred["cluster"].shift(1)
        df_pred["_changed"] = (df_pred["cluster"] != df_pred["_cluster_prev"]).astype(int)
        df_trans = df_pred.groupby(group_key, dropna=False)["_changed"].sum().rename("n_transitions").reset_index()
        df_trans.to_csv(p_trans, index=False, encoding="utf-8")
    else:
        # still write empty files for consistency
        pd.DataFrame().to_csv(p_density, index=False, encoding="utf-8")
        pd.DataFrame().to_csv(p_trans, index=False, encoding="utf-8")

    # Params
    p_npz = out / "gmm_params.npz"
    np.savez_compressed(p_npz, weights=res.gmm.weights_, means=res.gmm.means_, covariances=res.gmm.covariances_)

    # Plots (optional)
    if cfg.plots and "global_step" in df_pred.columns:
        x = df_pred["global_step"].to_numpy()
        yH = df_pred["entropy"].to_numpy()
        yC = df_pred["cluster"].to_numpy()

        plt.figure(figsize=(10, 3))
        plt.plot(x, yH)
        plt.xlabel("global_step")
        plt.ylabel("entropy")
        plt.title("Uncertainty (entropy) over time")
        _save_fig(out / "fig_entropy_timeline.png")

        plt.figure(figsize=(10, 3))
        plt.plot(x, yC)
        plt.xlabel("global_step")
        plt.ylabel("cluster")
        plt.title("Latent state (cluster) over time")
        _save_fig(out / "fig_cluster_timeline.png")

        plt.figure(figsize=(10, 3))
        plt.plot(x, yH)
        plt.scatter(x[idx_tp], yH[idx_tp])
        plt.xlabel("global_step")
        plt.ylabel("entropy")
        plt.title(f"Turning points (Top {cfg.top_k}) on entropy timeline")
        _save_fig(out / "fig_turning_points.png")

    return {
        "events_all": p_events,
        "preds_all": p_preds,
        "boundary_topK": p_tp,
        "boundary_context": p_ctx,
        "scene_turning_density": p_density,
        "scene_cluster_transitions": p_trans,
        "gmm_params": p_npz,
    }
