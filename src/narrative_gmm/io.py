from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
import json
import numpy as np
import pandas as pd

@dataclass
class LoadedData:
    df_events: pd.DataFrame
    X: np.ndarray

def _get(d: dict, k: str, default: float = 0.0) -> float:
    v = d.get(k, default)
    try:
        return float(v)
    except Exception:
        return float(default)

def _infer_feature_spec(ch: dict) -> dict:
    fs = ch.get("feature_spec", {}) or {}
    # fallback defaults (Ulysses_fixed.json seems to use these)
    return {
        "mode_keys": fs.get("mode_keys", ["perception","inner_speech","memory","imagination","reasoning","emotion","dialogue","quotation"]),
        "cause_keys": fs.get("cause_keys", ["sensory","memory","emotion_drive","linguistic","social","physio","intertext","goal_task"]),
        "style_keys": fs.get("style_keys", ["syntax_break","ellipsis","intertext_density","language_shift","phonetic_play"]),
    }

def load_events_json(json_path: str | Path) -> LoadedData:
    """
    Load Ulysses_fixed.json-like structure:
      doc[] where each doc has chapter_meta + feature_spec + time_series_data[]
    Builds X by flattening:
      mode + cause_weights + anchor_strength + place_vec + myth_vec + style
    """
    path = Path(json_path)
    obj = json.loads(path.read_text(encoding="utf-8"))

    rows = []
    X_rows = []

    for doc in obj:
        meta = doc.get("chapter_meta", {}) or {}
        doc_id = doc.get("doc_id", "")
        episode = meta.get("episode", doc.get("episode", None))
        work = meta.get("work", "Ulysses")
        spec = _infer_feature_spec(doc)

        for p in doc.get("time_series_data", []):
            mode = p.get("mode", {}) or {}
            cause = p.get("cause_weights", {}) or {}
            style = p.get("style", {}) or {}
            anchor = p.get("anchor", {}) or {}
            place_vec = p.get("place_vec", []) or []
            myth_vec = p.get("myth_vec", []) or []

            x = []
            x += [_get(mode, k) for k in spec["mode_keys"]]
            x += [_get(cause, k) for k in spec["cause_keys"]]
            x += [_get(anchor, "anchor_strength", default=0.0)]
            x += [float(v) for v in place_vec]
            x += [float(v) for v in myth_vec]
            x += [_get(style, k) for k in spec["style_keys"]]

            rows.append({
                "doc_id": doc_id,
                "work": work,
                "episode": episode,
                "global_step": p.get("global_step"),
                "span_text_en": p.get("span_text_en", ""),
                "span_text_ja": p.get("span_text_ja", ""),
                "evidence_en": p.get("evidence_en", p.get("evidence", "")),
                "evidence_ja": p.get("evidence_ja", ""),
                "transition_type": p.get("transition_type", ""),
                "time_anchor": anchor.get("time_anchor", ""),
                "place_anchor": anchor.get("place_anchor", ""),
                "anchor_strength": anchor.get("anchor_strength", None),
                "status": p.get("status", "unlabeled"),
                "label": p.get("label", None),
            })
            X_rows.append(np.asarray(x, dtype=float))

    df = pd.DataFrame(rows)
    df = df.dropna(subset=["global_step"]).copy()
    df["global_step"] = df["global_step"].astype(int)
    df = df.sort_values(["doc_id", "global_step"]).reset_index(drop=True)

    if len(X_rows) == 0:
        raise ValueError("No time_series_data found in JSON.")
    X = np.vstack(X_rows)
    return LoadedData(df_events=df, X=X)

def load_events_csv(csv_path: str | Path, feature_prefixes: tuple[str, ...] = ("mode_", "cause_", "place_", "myth_", "style_", "anchor_")) -> LoadedData:
    """
    Load a flattened feature CSV (like ulysses_stream.csv).
    Feature columns are auto-selected by prefixes.
    """
    path = Path(csv_path)
    df = pd.read_csv(path)
    # choose features
    feat_cols = [c for c in df.columns if any(c.startswith(p) for p in feature_prefixes)]
    if not feat_cols:
        raise ValueError(f"No feature columns found by prefixes={feature_prefixes}. columns(head)={list(df.columns)[:50]}")
    X = df[feat_cols].to_numpy(dtype=float)
    # try to enforce a time axis
    if "global_step" in df.columns:
        df = df.sort_values("global_step").reset_index(drop=True)
    return LoadedData(df_events=df, X=X)
