"""Sample-efficiency summary (Claim 3 of hypothesis.md v2).

Reads `results/sample_efficiency/{classical,qlnn}_h{3}_pct{10,25,50,100}/`
and produces:

  results/sample_efficiency/sample_efficiency_table.md       paper-style
  results/sample_efficiency/sample_efficiency_table.json     parsable
  results/sample_efficiency/sample_efficiency_curve.png      log(n_train) vs MAE

Acceptance threshold (hypothesis.md v2, Claim 3): QLNN reaches X at <= 50%
data while classical needs > 50%, where X = classical_H4 100%-data test MAE.

Usage:
    .venv/bin/python scripts/summarize_sample_efficiency.py
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
ROOT = REPO / "results" / "sample_efficiency"
FRACTIONS = [10, 25, 50, 100]


def _read_summary(run_dir: Path) -> dict | None:
    p = run_dir / "seeds_summary.json"
    if not p.exists():
        return None
    with p.open() as f:
        return json.load(f)


def _fmt_mean_ci(v: dict | None) -> str:
    if v is None:
        return "—"
    mean = v["mean"]
    ci = v.get("ci95_half_width", v.get("std"))
    return f"{mean:.4f} ± {ci:.4f}"


def main() -> None:
    if not ROOT.exists():
        print(f"no results at {ROOT}; nothing to summarize", file=sys.stderr)
        sys.exit(1)

    rows: dict[str, dict[int, dict | None]] = {"classical": {}, "qlnn": {}}
    n_train_windows: dict[str, dict[int, int | None]] = {"classical": {}, "qlnn": {}}

    for pct in FRACTIONS:
        c_dir = ROOT / f"classical_h4_h3_pct{pct}"
        q_dir = ROOT / f"qlnn_h3_pct{pct}"
        c_sum = _read_summary(c_dir)
        q_sum = _read_summary(q_dir)
        rows["classical"][pct] = c_sum
        rows["qlnn"][pct] = q_sum
        # Pull window count from protocol.json if present.
        for stack, run_dir in [("classical", c_dir), ("qlnn", q_dir)]:
            p_file = run_dir / "protocol.json"
            if p_file.exists():
                with p_file.open() as f:
                    n_train_windows[stack][pct] = json.load(f).get("n_train_windows")
            else:
                n_train_windows[stack][pct] = None

    # ---- Markdown table ----
    md: list[str] = []
    md.append("# Sample-efficiency sweep (Claim 3 / Step 6)")
    md.append("")
    md.append("Test MAE on the locked h=3 evaluation, fraction of training")
    md.append("windows truncated chronologically from the start. mean ± 95% CI.")
    md.append("")
    md.append("| Stack | 10% | 25% | 50% | 100% |")
    md.append("|---|---|---|---|---|")
    for stack in ("classical", "qlnn"):
        cells = []
        for pct in FRACTIONS:
            s = rows[stack][pct]
            if s is None:
                cells.append("—")
            else:
                mae = s["test"].get("mae_raw")
                cells.append(_fmt_mean_ci(mae))
        label = "Classical H=4" if stack == "classical" else "QLNN"
        md.append(f"| {label} | " + " | ".join(cells) + " |")

    md.append("")
    md.append("Window counts (training only):")
    md.append("")
    md.append("| Stack | 10% | 25% | 50% | 100% |")
    md.append("|---|---|---|---|---|")
    for stack in ("classical", "qlnn"):
        cells = [str(n_train_windows[stack][pct]) if n_train_windows[stack][pct] is not None else "—"
                 for pct in FRACTIONS]
        label = "Classical H=4" if stack == "classical" else "QLNN"
        md.append(f"| {label} | " + " | ".join(cells) + " |")

    # ---- Claim 3 verdict ----
    target_run = rows["classical"].get(100)
    if target_run is not None:
        target_mae = target_run["test"]["mae_raw"]["mean"]
        md.append("")
        md.append(f"### Claim 3 verdict")
        md.append("")
        md.append(f"Target X = classical H=4 test MAE at 100% data = **{target_mae:.4f}**.")
        for pct in FRACTIONS:
            q = rows["qlnn"][pct]
            c = rows["classical"][pct]
            q_mae = q["test"]["mae_raw"]["mean"] if q else None
            c_mae = c["test"]["mae_raw"]["mean"] if c else None
            md.append(
                f"- {pct}%: classical={c_mae:.4f} | QLNN={q_mae:.4f} | "
                f"QLNN reaches target ({q_mae:.4f} <= {target_mae:.4f}): "
                f"{'YES' if q_mae is not None and q_mae <= target_mae else 'NO'} | "
                f"classical reaches target: "
                f"{'YES' if c_mae is not None and c_mae <= target_mae else 'NO'}"
            )

    out_md = ROOT / "sample_efficiency_table.md"
    out_md.write_text("\n".join(md) + "\n")
    print("\n".join(md))
    print(f"\nwrote {out_md}")

    # ---- JSON blob ----
    out_json = ROOT / "sample_efficiency_table.json"
    out_json.write_text(json.dumps({
        "fractions": FRACTIONS,
        "rows": rows,
        "n_train_windows": n_train_windows,
    }, indent=2, default=str) + "\n")
    print(f"wrote {out_json}")

    # ---- Optional plot ----
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 5))
        for stack, marker, label in (("classical", "o", "Classical H=4"),
                                      ("qlnn", "s", "QLNN")):
            xs = []
            means = []
            cis = []
            for pct in FRACTIONS:
                s = rows[stack][pct]
                if s is None:
                    continue
                mae = s["test"].get("mae_raw")
                if not mae:
                    continue
                xs.append(n_train_windows[stack][pct])
                means.append(mae["mean"])
                cis.append(mae.get("ci95_half_width", mae.get("std", 0)))
            if xs:
                xs = list(xs); means = list(means); cis = list(cis)
                ax.errorbar(xs, means, yerr=cis, marker=marker, label=label, capsize=4)
        ax.set_xscale("log")
        ax.set_xlabel("Training windows (log scale)")
        ax.set_ylabel("Test MAE_raw (95% CI)")
        ax.set_title("Sample efficiency at h=3")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        out_png = ROOT / "sample_efficiency_curve.png"
        fig.savefig(out_png, dpi=150)
        plt.close(fig)
        print(f"wrote {out_png}")
    except ImportError:
        print("(matplotlib not available; skipping plot)")


if __name__ == "__main__":
    main()
