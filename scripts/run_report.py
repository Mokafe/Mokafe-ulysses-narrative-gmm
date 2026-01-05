#!/usr/bin/env python3
from __future__ import annotations
import argparse
from pathlib import Path

from narrative_gmm.report import build_report, ReportConfig

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input .json or .csv")
    ap.add_argument("--out", default="outputs", help="Output directory")
    ap.add_argument("--k", type=int, default=2, help="Number of GMM components")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--alpha", type=float, default=20.0, help="Anchor pull strength")
    ap.add_argument("--top_k", type=int, default=20, help="Top K turning points")
    ap.add_argument("--context_w", type=int, default=3, help="Context window (+/-)")
    ap.add_argument("--plots", action="store_true", help="Save plots as PNG")
    args = ap.parse_args()

    cfg = ReportConfig(
        n_components=args.k,
        seed=args.seed,
        alpha=args.alpha,
        top_k=args.top_k,
        context_w=args.context_w,
        plots=args.plots,
    )
    out = build_report(args.input, args.out, cfg)
    print("[Saved]")
    for k, v in out.items():
        print(f"- {k}: {Path(v).as_posix()}")

if __name__ == "__main__":
    main()
