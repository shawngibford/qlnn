"""Summarize the param-matched classical Liquid-ODE sweep.

Reads `results/param_sweep/euler_h3_hidden*` run dirs (produced by
`scripts/run_param_sweep.sh`), computes each model's parameter count from its
`best_state.pt`, and emits:

    results/param_sweep/param_sweep_table.md   # paper-style summary table
    results/param_sweep/param_pareto.csv       # long-form per-seed CSV
    results/param_sweep/param_pareto.png       # MAE vs params Pareto (if matplotlib)

Optional: pass --qlnn-run PATH to overlay the QLNN's mean MAE point.

Phase C, Tier 2 #2.1 — close the 20x param-count gap between classical
Liquid-ODE (~1601) and the QLNN (~100).
"""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import torch


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SWEEP_DIR = REPO_ROOT / "results" / "param_sweep"
HIDDEN_SIZES = [2, 4, 8, 16, 32]


def _load_json(p: Path) -> Any:
    with p.open("r") as f:
        return json.load(f)


def _param_count_from_state(state_path: Path) -> int:
    """Sum numel over all tensors in a torch state_dict checkpoint."""
    state = torch.load(state_path, map_location="cpu", weights_only=True)
    if not isinstance(state, dict):
        raise ValueError(f"unexpected checkpoint format at {state_path}: {type(state)}")
    return int(sum(int(t.numel()) for t in state.values() if isinstance(t, torch.Tensor)))


def _seed_dirs(run_dir: Path) -> list[Path]:
    return sorted([p for p in run_dir.glob("seed_*") if p.is_dir()],
                  key=lambda p: int(p.name.split("_", 1)[1]))


def _mean_std(xs: list[float]) -> tuple[float, float]:
    n = len(xs)
    if n == 0:
        return float("nan"), float("nan")
    m = sum(xs) / n
    if n == 1:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var)


def _collect_run(run_dir: Path) -> dict[str, Any] | None:
    """Pull params and per-seed test metrics from one run dir.

    Returns None if the run dir is missing / incomplete.
    """
    summary_path = run_dir / "seeds_summary.json"
    if not summary_path.exists():
        return None
    seed_dirs = _seed_dirs(run_dir)
    if not seed_dirs:
        return None

    # Param count from the first available best_state.pt — model arch is
    # identical across seeds, so any one is canonical.
    params: int | None = None
    for sd in seed_dirs:
        sp = sd / "best_state.pt"
        if sp.exists():
            params = _param_count_from_state(sp)
            break
    if params is None:
        return None

    per_seed: list[dict[str, float]] = []
    for sd in seed_dirs:
        mp = sd / "metrics.json"
        if not mp.exists():
            continue
        m = _load_json(mp)
        test = m.get("test", {})
        per_seed.append({
            "seed": int(sd.name.split("_", 1)[1]),
            "mae_raw": float(test.get("mae_raw", float("nan"))),
            "rmse_raw": float(test.get("rmse_raw", float("nan"))),
            "r2_raw": float(test.get("r2_raw", float("nan"))),
            "delta_r2_raw": float(test.get("delta_r2_raw", float("nan"))),
        })
    if not per_seed:
        return None

    return {"run_dir": str(run_dir), "params": params, "per_seed": per_seed}


def _build_qlnn_entry(qlnn_run: Path) -> dict[str, Any] | None:
    """Best-effort QLNN extraction — same shape as a sweep entry."""
    if not qlnn_run.exists():
        return None
    entry = _collect_run(qlnn_run)
    if entry is None:
        # Fallback: read seeds_summary.json directly without best_state.pt.
        sp = qlnn_run / "seeds_summary.json"
        if not sp.exists():
            return None
        summary = _load_json(sp)
        test = summary.get("test", {})
        def _g(k):
            v = test.get(k, {})
            return float(v.get("mean", float("nan")))
        per_seed = [{
            "seed": -1,
            "mae_raw": _g("mae_raw"),
            "rmse_raw": _g("rmse_raw"),
            "r2_raw": _g("r2_raw"),
            "delta_r2_raw": _g("delta_r2_raw"),
        }]
        return {"run_dir": str(qlnn_run), "params": -1, "per_seed": per_seed}
    return entry


def _emit_csv(out_path: Path, rows: list[tuple[str, int, int, dict[str, float]]]) -> None:
    lines = ["label,hidden_size,params,seed,test_mae_raw,test_rmse_raw,test_r2_raw,test_delta_r2_raw"]
    for label, hidden_size, params, m in rows:
        lines.append(
            f"{label},{hidden_size},{params},{m['seed']},"
            f"{m['mae_raw']:.6f},{m['rmse_raw']:.6f},{m['r2_raw']:.6f},{m['delta_r2_raw']:.6f}"
        )
    out_path.write_text("\n".join(lines) + "\n")


def _emit_table(out_path: Path, summary_rows: list[dict[str, Any]]) -> None:
    header = (
        "| Model | hidden_size | params | test MAE_raw (mean ± std) | "
        "test R²_raw (mean ± std) | test ΔR²_raw (mean ± std) |\n"
        "|---|---|---|---|---|---|"
    )
    body: list[str] = []
    for r in summary_rows:
        body.append(
            f"| {r['label']} | {r['hidden_size']} | {r['params']} | "
            f"{r['mae_mean']:.4f} ± {r['mae_std']:.4f} | "
            f"{r['r2_mean']:.4f} ± {r['r2_std']:.4f} | "
            f"{r['delta_r2_mean']:.4f} ± {r['delta_r2_std']:.4f} |"
        )
    out_path.write_text(
        "# Param-matched classical Liquid-ODE sweep (h=3)\n\n"
        "Tier 2 #2.1 from the peer-review swarm: classical hidden_size ∈ {2,4,8,16,32} "
        "vs the QLNN (~100 params). All rows use the same h=3 protocol "
        "(`configs/param_sweep/*.yaml`), 5 seeds each.\n\n"
        + header + "\n" + "\n".join(body) + "\n"
    )


def _maybe_plot(png_path: Path, summary_rows: list[dict[str, Any]]) -> str | None:
    try:
        import matplotlib  # type: ignore
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt  # type: ignore
    except ImportError:
        return "matplotlib not installed; skipping param_pareto.png"

    classical = [r for r in summary_rows if r["label"] == "classical"]
    qlnn = [r for r in summary_rows if r["label"] == "qlnn"]
    if not classical:
        return "no classical rows — skipping plot"

    xs = [r["params"] for r in classical]
    ys = [r["mae_mean"] for r in classical]
    err = [r["mae_std"] for r in classical]

    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.errorbar(xs, ys, yerr=err, fmt="o-", color="C0", capsize=3,
                label="Classical Liquid-ODE")
    for r in classical:
        ax.annotate(f"H={r['hidden_size']}", (r["params"], r["mae_mean"]),
                    xytext=(4, 4), textcoords="offset points", fontsize=9)
    if qlnn:
        q = qlnn[0]
        ax.errorbar([q["params"]], [q["mae_mean"]], yerr=[q["mae_std"]],
                    fmt="s", color="C3", capsize=3, label="QLNN")
        ax.annotate("QLNN", (q["params"], q["mae_mean"]),
                    xytext=(4, -10), textcoords="offset points", fontsize=9, color="C3")
    ax.set_xscale("log")
    ax.set_xlabel("trainable params (log)")
    ax.set_ylabel("test MAE_raw")
    ax.set_title("Param-matched Pareto (h=3, 5 seeds)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(png_path, dpi=150)
    plt.close(fig)
    return None


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the param-matched sweep.")
    parser.add_argument("--sweep-dir", type=Path, default=DEFAULT_SWEEP_DIR,
                        help="Directory containing euler_h3_hidden* run dirs.")
    parser.add_argument("--qlnn-run", type=Path, default=None,
                        help="Optional QLNN run dir to overlay (e.g. results/qlnn_hybrid).")
    args = parser.parse_args()

    sweep_dir: Path = args.sweep_dir
    if not sweep_dir.exists():
        raise SystemExit(f"sweep dir not found: {sweep_dir} — run scripts/run_param_sweep.sh first.")

    csv_rows: list[tuple[str, int, int, dict[str, float]]] = []
    summary_rows: list[dict[str, Any]] = []

    for H in HIDDEN_SIZES:
        run_dir = sweep_dir / f"euler_h3_hidden{H}"
        entry = _collect_run(run_dir)
        if entry is None:
            print(f"[warn] skipping H={H}: no results at {run_dir}")
            continue
        params = entry["params"]
        per_seed = entry["per_seed"]
        for s in per_seed:
            csv_rows.append(("classical", H, params, s))
        maes = [s["mae_raw"] for s in per_seed]
        r2s = [s["r2_raw"] for s in per_seed]
        dr2s = [s["delta_r2_raw"] for s in per_seed]
        mae_m, mae_s = _mean_std(maes)
        r2_m, r2_s = _mean_std(r2s)
        dr2_m, dr2_s = _mean_std(dr2s)
        summary_rows.append({
            "label": "classical", "hidden_size": H, "params": params,
            "mae_mean": mae_m, "mae_std": mae_s,
            "r2_mean": r2_m, "r2_std": r2_s,
            "delta_r2_mean": dr2_m, "delta_r2_std": dr2_s,
        })

    if args.qlnn_run is not None:
        q_entry = _build_qlnn_entry(args.qlnn_run)
        if q_entry is None:
            print(f"[warn] --qlnn-run given but no results found at {args.qlnn_run}")
        else:
            qpar = q_entry["params"]
            qps = q_entry["per_seed"]
            for s in qps:
                csv_rows.append(("qlnn", -1, qpar, s))
            maes = [s["mae_raw"] for s in qps]
            r2s = [s["r2_raw"] for s in qps]
            dr2s = [s["delta_r2_raw"] for s in qps]
            mae_m, mae_s = _mean_std(maes)
            r2_m, r2_s = _mean_std(r2s)
            dr2_m, dr2_s = _mean_std(dr2s)
            summary_rows.append({
                "label": "qlnn", "hidden_size": "—", "params": qpar,
                "mae_mean": mae_m, "mae_std": mae_s,
                "r2_mean": r2_m, "r2_std": r2_s,
                "delta_r2_mean": dr2_m, "delta_r2_std": dr2_s,
            })

    if not summary_rows:
        raise SystemExit("no runs collected — nothing to summarize.")

    sweep_dir.mkdir(parents=True, exist_ok=True)
    csv_path = sweep_dir / "param_pareto.csv"
    table_path = sweep_dir / "param_sweep_table.md"
    png_path = sweep_dir / "param_pareto.png"
    _emit_csv(csv_path, csv_rows)
    _emit_table(table_path, summary_rows)
    plot_note = _maybe_plot(png_path, summary_rows)

    print(f"wrote: {table_path}")
    print(f"wrote: {csv_path}")
    if plot_note:
        print(f"plot:  {plot_note}")
    else:
        print(f"wrote: {png_path}")

    print()
    print(table_path.read_text())


if __name__ == "__main__":
    main()
