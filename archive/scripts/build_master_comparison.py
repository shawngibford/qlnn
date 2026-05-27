"""Master all-vs-all comparison — every classical + quantum configuration
in one ranked table on a shared footing.

Unifies four on-disk sources (no new compute):
  - results/baseline_lock.json                  classical sweep + QLNN ref
  - results/circuit_search/circuit_search_table.json   prior search
  - results/option_b/option_b_table.json        Option-B circuit×regime
  - results/baseline_lock.json gate values      G1/G2 + SE no-regression

Emits:
  results/master_comparison.json   machine-readable, every row
  results/master_comparison.md     reviewer-facing ranked table
  results/master_comparison.csv    long-form

Each row carries: family/source, params (best-effort), test MAE mean±σ,
test R², σ-ratio vs classical_H4, penalized objective (where σ known),
and G1/G2 verdicts. Sorted by test MAE.

Gracefully no-ops sources that are absent so it is safe to run before
the Option-B sweep finishes (it just won't include those rows yet).

Usage:
    python scripts/build_master_comparison.py
"""
from __future__ import annotations

import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "results" / "baseline_lock.json"


def _load(p: Path):
    with p.open() as f:
        return json.load(f)


def _clean(x):
    """NaN/None → None (single-seed runs have an undefined σ; show '—')."""
    if x is None:
        return None
    try:
        if x != x:  # NaN
            return None
    except TypeError:
        return None
    return x


def _row(source, name, params, mae, sigma, r2, g1, g2, sigma_cl, penalized=None):
    mae = _clean(mae)
    sigma = _clean(sigma)
    r2 = _clean(r2)
    ratio = (sigma_cl / sigma) if (sigma and sigma > 0) else None
    return {
        "source": source, "name": name, "params": params,
        "test_mae_mean": mae, "test_mae_sigma": sigma, "test_r2": r2,
        "sigma_ratio_vs_classical_H4": round(ratio, 3) if ratio else None,
        "penalized": penalized,
        "G1_accuracy_pass": (mae is not None and mae < g1),
        "G2_reproducibility_pass": (sigma is not None and sigma <= g2),
    }


def main() -> None:
    if not LOCK.exists():
        print(f"missing {LOCK} — generate the baseline lock first")
        return
    lock = _load(LOCK)
    g1 = lock["classical"]["matched_param_H4"]["test_mae"]["mean"]
    sigma_cl = lock["claim1_sigma_ratio"]["sigma_classical_H4"]
    g2 = 0.5 * sigma_cl

    rows: list[dict] = []

    # --- 1. Classical param sweep + QLNN reference (from the lock) --------
    for tag, key, params in [
        ("classical", "matched_param_H4", 90),
        ("classical", "best_param_sweep_cell_H2", 42),
    ]:
        c = lock["classical"][key]["test_mae"]
        rows.append(_row("classical_sweep", f"classical_{key}", params,
                         c["mean"], c["std"], None, g1, g2, sigma_cl))
    qr = lock["qlnn_reference"]["test_mae"]
    rows.append(_row("qlnn_reference", "data_reuploading_4q3L_ring_rx", 114,
                     qr["mean"], qr["std"], None, g1, g2, sigma_cl))

    # --- 2. Prior circuit search ----------------------------------------
    pj = ROOT / "results" / "circuit_search" / "circuit_search_table.json"
    if pj.exists():
        for r in _load(pj):
            mae = r["test_mae_raw"]["mean"]
            sd = r["test_mae_raw"].get("std")
            r2 = r["test_r2_raw"]["mean"]
            rows.append(_row(
                f"prior_search/{r['axis']}",
                f"{r['ansatz_name']}_{r['run']}",
                None, mae, sd, r2, g1, g2, sigma_cl))

    # --- 3. Option-B circuit × regime -----------------------------------
    oj = ROOT / "results" / "option_b" / "option_b_table.json"
    if oj.exists():
        for r in _load(oj):
            rows.append(_row(
                "option_b",
                f"{r['circuit']}__{r['regime']}",
                None, r["test_mae_mean"], r["test_mae_sigma"],
                None, g1, g2, sigma_cl, penalized=r.get("penalized")))

    # Rank by test MAE (Nones last).
    rows.sort(key=lambda r: (r["test_mae_mean"] is None,
                             r["test_mae_mean"] if r["test_mae_mean"]
                             is not None else 9e9))

    out_json = ROOT / "results" / "master_comparison.json"
    out_json.write_text(json.dumps(rows, indent=2) + "\n")

    md = ["# Master comparison — all classical + quantum configurations\n"]
    md.append(f"G1 accuracy bar = classical H=4 MAE **{g1:.4f}**; "
              f"G2 σ gate = **{g2:.5f}** (½·σ_classical_H4 = "
              f"½·{sigma_cl:.5f}). Ranked by test MAE. "
              f"σ-ratio = σ_classical_H4 / σ_row (Claim-1 needs ≥ 2×).\n")
    md.append("| Rank | Source | Config | Params | Test MAE | σ | σ-ratio | "
              "Test R² | G1 | G2 |")
    md.append("|---|---|---|---|---|---|---|---|---|---|")
    for i, r in enumerate(rows, 1):
        def f(x, nd=4):
            return "—" if x is None else f"{x:.{nd}f}"
        md.append(
            f"| {i} | {r['source']} | {r['name']} | "
            f"{r['params'] if r['params'] is not None else '—'} | "
            f"{f(r['test_mae_mean'])} | {f(r['test_mae_sigma'],5)} | "
            f"{f(r['sigma_ratio_vs_classical_H4'],2)} | {f(r['test_r2'],3)} | "
            f"{'✅' if r['G1_accuracy_pass'] else '❌'} | "
            f"{'✅' if r['G2_reproducibility_pass'] else '❌'} |")
    both = [r for r in rows
            if r["G1_accuracy_pass"] and r["G2_reproducibility_pass"]]
    md.append("")
    md.append(f"**{len(both)} configuration(s) pass BOTH G1 (accuracy) "
              f"and G2 (reproducibility)** — i.e. the Option-B 'best for "
              f"all' target at the proxy level.")
    for r in both:
        md.append(f"- `{r['source']}/{r['name']}` "
                  f"(MAE {r['test_mae_mean']:.4f}, σ {r['test_mae_sigma']:.5f})")
    (ROOT / "results" / "master_comparison.md").write_text(
        "\n".join(md) + "\n")

    with (ROOT / "results" / "master_comparison.csv").open(
            "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["rank", "source", "name", "params", "test_mae_mean",
                    "test_mae_sigma", "sigma_ratio_vs_classical_H4",
                    "test_r2", "penalized", "G1_accuracy_pass",
                    "G2_reproducibility_pass"])
        for i, r in enumerate(rows, 1):
            w.writerow([i, r["source"], r["name"], r["params"],
                        r["test_mae_mean"], r["test_mae_sigma"],
                        r["sigma_ratio_vs_classical_H4"], r["test_r2"],
                        r["penalized"], r["G1_accuracy_pass"],
                        r["G2_reproducibility_pass"]])

    print(f"wrote results/master_comparison.{{json,md,csv}} "
          f"({len(rows)} rows, {len(both)} pass G1+G2)")


if __name__ == "__main__":
    main()
