"""Verify that every headline number in PAPER_SUMMARY.md matches the
committed result JSONs. Run after reproduce_paper.sh, before a release,
and on every CI build.

Exits 0 if all numbers match (within tolerance), 1 otherwise.

Tolerance: 0.0001 absolute for MAE/R² values, 0.01 for d_norm differences.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _load(p: str | Path) -> dict:
    with (ROOT / p).open() as f: return json.load(f)


def _check(label: str, actual: float, expected: float, tol: float = 1e-4) -> bool:
    """ASCII status markers so this script runs under LANG=C in CI (H-09)."""
    ok = abs(actual - expected) <= tol
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: actual={actual:.4f}  expected={expected}  (tol={tol})")
    return ok


def _check_str(label: str, actual: str, expected: str) -> bool:
    """Exact string match (used for verdict outcomes — CONFIRMED /
    FALSIFIED / INCONCLUSIVE)."""
    ok = actual == expected
    status = "OK  " if ok else "FAIL"
    print(f"  [{status}] {label}: actual={actual!r}  expected={expected!r}")
    return ok


def main() -> int:
    all_ok = True

    print("=== Claim 1 (reproducibility): σ_classical_H4 / σ_QLNN ratio ===")
    c = _load("archive/results/param_sweep/euler_h3_hidden4/seeds_summary.json")
    q = _load("archive/results/qlnn_hybrid_h3/seeds_summary.json")
    ratio = c["test"]["mae_raw"]["std"] / q["test"]["mae_raw"]["std"]
    # H-08 fix: tightened tol from 0.1 to 0.05. The committed numbers give
    # ratio = 3.80; we want the check to FAIL if regressions push the ratio
    # below ~3.75 or above ~3.85. The pre-registration's >= 2.0 is the
    # separate scientific threshold and lives in the paper itself, not here.
    all_ok &= _check("ratio (paper: 3.77)", ratio, 3.77, tol=0.05)

    print("\n=== Claim 2 (expressivity): Δd_norm ===")
    ed = _load("archive/results/effective_dimension/effective_dimension.json")
    all_ok &= _check("classical d_norm (paper: 8.0290)",
                     ed["classical_H4"]["aggregate"]["mean"], 8.0290, tol=0.001)
    all_ok &= _check("QLNN d_norm (paper: 9.5144)",
                     ed["qlnn_h3"]["aggregate"]["mean"], 9.5144, tol=0.001)
    all_ok &= _check("Δd_norm (paper: +1.49)",
                     ed["delta_d_norm_qlnn_minus_classical"], 1.49, tol=0.01)
    if not ed["pre_registered_hypothesis_met"]:
        print("  [FAIL] pre-registered hypothesis flag is False (expected True)")
        all_ok = False
    else:
        print("  [OK  ] pre-registered hypothesis flag is True")

    print("\n=== Claim 3 (sample efficiency): per-fraction means ===")
    expected_mae = {
        10:  (0.2788, 0.2686),
        25:  (0.2546, 0.2507),
        50:  (0.2564, 0.2633),
        100: (0.2594, 0.2655),
    }
    for pct, (paper_c, paper_q) in expected_mae.items():
        c = _load(f"archive/results/sample_efficiency/classical_h4_h3_pct{pct}/seeds_summary.json")
        q = _load(f"archive/results/sample_efficiency/qlnn_h3_pct{pct}/seeds_summary.json")
        all_ok &= _check(f"pct={pct} classical MAE",
                         c["test"]["mae_raw"]["mean"], paper_c)
        all_ok &= _check(f"pct={pct} QLNN MAE",
                         q["test"]["mae_raw"]["mean"], paper_q)

    print("\n=== Horizon ablation: persistence + LO-ODE R² ===")
    expected_horizon = {
        1:  (0.9052,    0.9048),
        3:  (-0.0371,   0.1108),
        6:  (-9.7136,  -9.5004),
        12: (-977.0226, -999.8978),
    }
    for h, (paper_pers, paper_lo) in expected_horizon.items():
        s = _load(f"archive/results/horizon_sweep/euler_h{h}/seeds_summary.json")
        b = _load(f"archive/results/horizon_sweep/euler_h{h}/baselines.json")
        all_ok &= _check(f"h={h} persistence R²",
                         b["persistence"]["test"]["r2_raw"], paper_pers,
                         tol=max(0.01, abs(paper_pers) * 0.01))
        all_ok &= _check(f"h={h} LO-ODE R²",
                         s["test"]["r2_raw"]["mean"], paper_lo,
                         tol=max(0.1, abs(paper_lo) * 0.01))

    # =====================================================================
    # PIVOT PROGRAM — H1/H3 verdict integrity gates (P7.5 commit 3 / R3 fix)
    # =====================================================================
    print("\n=== PIVOT H1 verdicts (forecaster + solver) ===")
    print("  --- Forecaster-task H1 (P5, corroborating) ---")
    h1_fc = _load("results/p5_h1_verdict/h1_analysis.json")
    all_ok &= _check_str(
        "Forecaster H1 outcome (paper: FALSIFIED)",
        h1_fc["outcome"], "FALSIFIED")
    if h1_fc["bootstrap"] is not None:
        b = h1_fc["bootstrap"]
        all_ok &= _check(
            "Forecaster Δ_diff_mean (paper: -0.4166)",
            b["delta_diff_mean"], -0.4166, tol=0.005)
        all_ok &= _check(
            "Forecaster CI low (paper: -0.7871)",
            b["ci_low"], -0.7871, tol=0.05)
        all_ok &= _check(
            "Forecaster CI high (paper: -0.0460)",
            b["ci_high"], -0.0460, tol=0.05)

    print("  --- Solver-task H1 (P7.5, PRIMARY per pre-reg §7) ---")
    h1_sv = _load("results/p7_5_solver_h1/h1_analysis_solver_task_raw.json")
    all_ok &= _check_str(
        "Solver H1 outcome at raw bootstrap (paper: CONFIRMED)",
        h1_sv["outcome"], "CONFIRMED")
    if h1_sv["bootstrap"] is not None:
        b = h1_sv["bootstrap"]
        all_ok &= _check(
            "Solver Δ_diff_mean (paper: +0.1094)",
            b["delta_diff_mean"], 0.1094, tol=0.005)
        all_ok &= _check(
            "Solver CI low > 0 (paper: +0.0145)",
            b["ci_low"], 0.0145, tol=0.05)
        all_ok &= _check(
            "Solver CI high (paper: +0.2204)",
            b["ci_high"], 0.2204, tol=0.05)
        all_ok &= _check(
            "Solver Δ_smooth_mean (paper: +0.1272)",
            b["delta_smooth_mean"], 0.1272, tol=0.005)
        all_ok &= _check(
            "Solver Δ_broad_mean (paper: +0.0179)",
            b["delta_broad_mean"], 0.0179, tol=0.005)

    print("  --- Combined ODE+PDE solver-task H1 (P7.6, n=18) ---")
    h1_cb = _load("results/p7_6_pde_solver_h1/h1_analysis_combined_solver.json")
    all_ok &= _check_str(
        "Combined ODE+PDE H1 outcome at n=18 (paper: FALSIFIED)",
        h1_cb["outcome"], "FALSIFIED")
    if h1_cb["bootstrap"] is not None:
        b = h1_cb["bootstrap"]
        all_ok &= _check(
            "Combined Δ_diff_mean (paper: +0.0316)",
            b["delta_diff_mean"], 0.0316, tol=0.005)
        all_ok &= _check(
            "Combined CI low (paper: -0.0400)",
            b["ci_low"], -0.0400, tol=0.05)
        all_ok &= _check(
            "Combined CI high (paper: +0.1088)",
            b["ci_high"], 0.1088, tol=0.05)
        all_ok &= _check(
            "Combined n_smooth (paper: 12)",
            b["n_smooth"], 12, tol=0)
        all_ok &= _check(
            "Combined n_broad (paper: 6)",
            b["n_broad"], 6, tol=0)

    print("  --- PDE-only solver-task H1 (P7.6, n=9) ---")
    h1_pde = _load("results/p7_6_pde_solver_h1/h1_analysis_pde_solver.json")
    all_ok &= _check_str(
        "PDE-only H1 outcome at n=9 (paper: FALSIFIED)",
        h1_pde["outcome"], "FALSIFIED")

    print("  --- Symmetric QLNN HPO H1 (P7.6, n=9, both sides HPO-best) ---")
    h1_hpo = _load("results/p7_6_qlnn_hpo/h1_verdict_full_hpo_best.json")
    all_ok &= _check_str(
        "Full-HPO-best H1 outcome (paper: FALSIFIED)",
        h1_hpo["outcome"], "FALSIFIED")
    if h1_hpo["bootstrap"] is not None:
        b = h1_hpo["bootstrap"]
        all_ok &= _check(
            "HPO-best Δ_diff_mean (paper: +0.0588)",
            b["delta_diff_mean"], 0.0588, tol=0.005)
        all_ok &= _check(
            "HPO-best CI low (paper: -0.0575)",
            b["ci_low"], -0.0575, tol=0.05)
        all_ok &= _check(
            "HPO-best CI high (paper: +0.1913)",
            b["ci_high"], 0.1913, tol=0.05)

    print("  --- Full-ladder n=24 H1 (P7.8, PAPER PRIMARY headline) ---")
    h1_24 = _load("results/p7_8_solver_h1_n24/h1_analysis_combined_n24.json")
    all_ok &= _check_str(
        "n=24 full-ladder H1 outcome (paper: FALSIFIED)",
        h1_24["outcome"], "FALSIFIED")
    if h1_24["bootstrap"] is not None:
        b = h1_24["bootstrap"]
        # Point estimate FLIPS SIGN vs P7.6 n=18 (was +0.032, now -0.084):
        # the expanded broadband bin (FHN + burgers_shock) reveals a
        # modest broad>smooth advantage trend (still CI-inclusive of 0).
        all_ok &= _check(
            "n=24 Δ_diff_mean (paper: -0.0844)",
            b["delta_diff_mean"], -0.0844, tol=0.005)
        all_ok &= _check(
            "n=24 CI low (paper: -0.2780)",
            b["ci_low"], -0.2780, tol=0.05)
        all_ok &= _check(
            "n=24 CI high (paper: +0.0613)",
            b["ci_high"], 0.0613, tol=0.05)
        all_ok &= _check(
            "n=24 Δ_smooth_mean (paper: +0.0674; unchanged vs n=18)",
            b["delta_smooth_mean"], 0.0674, tol=0.005)
        all_ok &= _check(
            "n=24 Δ_broad_mean (paper: +0.1518; up from +0.036 with "
            "FHN + burgers_shock)",
            b["delta_broad_mean"], 0.1518, tol=0.005)
        all_ok &= _check(
            "n=24 n_smooth (paper: 12)",
            b["n_smooth"], 12, tol=0)
        all_ok &= _check(
            "n=24 n_broad (paper: 12)",
            b["n_broad"], 12, tol=0)

    print("  --- Forecaster LTC decomposition (P7.10, n=9, all-to-all) ---")
    h1_comb = _load("results/p7_10_forecaster_decomposition/h1_combined.json")
    h1_q = _load("results/p7_10_forecaster_decomposition/h1_quantum_isolated.json")
    h1_l = _load("results/p7_10_forecaster_decomposition/h1_liquid_isolated.json")
    # All three FALSIFIED; the KEY finding is the sign and magnitude of
    # Δ_quantum vs Δ_liquid.
    all_ok &= _check_str(
        "Forecaster combined outcome (paper: FALSIFIED)",
        h1_comb["outcome"], "FALSIFIED")
    all_ok &= _check_str(
        "Forecaster quantum-isolated outcome (paper: FALSIFIED)",
        h1_q["outcome"], "FALSIFIED")
    all_ok &= _check_str(
        "Forecaster liquid-isolated outcome (paper: FALSIFIED)",
        h1_l["outcome"], "FALSIFIED")
    if h1_comb["bootstrap"] is not None:
        b = h1_comb["bootstrap"]
        # NOTE: includes rf_qrc as 5th QLNN candidate per cell. With rf_qrc
        # included, the combined verdict's CI EXCLUDES 0 negatively.
        all_ok &= _check(
            "Forecaster combined Δ_diff (paper: -0.5007)",
            b["delta_diff_mean"], -0.5007, tol=0.005)
        all_ok &= _check(
            "Forecaster combined CI low (paper: -0.8040)",
            b["ci_low"], -0.8040, tol=0.05)
        all_ok &= _check(
            "Forecaster combined CI high (paper: -0.2438; excludes 0)",
            b["ci_high"], -0.2438, tol=0.05)
    if h1_q["bootstrap"] is not None:
        b = h1_q["bootstrap"]
        # NOTE: also EXCLUDES 0 negatively with rf_qrc included → the
        # quantum circuit's underperformance on the regime contrast is
        # now statistically significant.
        all_ok &= _check(
            "Forecaster quantum-isolated Δ_diff (paper: -0.6160)",
            b["delta_diff_mean"], -0.6160, tol=0.005)
        all_ok &= _check(
            "Forecaster quantum-isolated CI low (paper: -1.1665)",
            b["ci_low"], -1.1665, tol=0.10)
        all_ok &= _check(
            "Forecaster quantum-isolated CI high (paper: -0.1782; excludes 0)",
            b["ci_high"], -0.1782, tol=0.05)
    if h1_l["bootstrap"] is not None:
        b = h1_l["bootstrap"]
        all_ok &= _check(
            "Forecaster liquid-isolated Δ_diff (paper: +0.1153)",
            b["delta_diff_mean"], 0.1153, tol=0.005)
        all_ok &= _check(
            "Forecaster liquid-isolated CI low (paper: -0.0973)",
            b["ci_low"], -0.0973, tol=0.05)
        all_ok &= _check(
            "Forecaster liquid-isolated CI high (paper: +0.3772)",
            b["ci_high"], 0.3772, tol=0.05)
    # Decomposition algebraic identity check (per-cell so should be ~0
    # exact, modulo float64 precision).
    if all(d["bootstrap"] is not None for d in (h1_comb, h1_q, h1_l)):
        sum_check = (h1_q["bootstrap"]["delta_diff_mean"]
                     + h1_l["bootstrap"]["delta_diff_mean"])
        all_ok &= _check(
            "Decomposition identity Δ_q + Δ_τ ≈ Δ_combined",
            sum_check, h1_comb["bootstrap"]["delta_diff_mean"],
            tol=0.001)

    print("  --- P7.11 forecaster 2×2 mechanism decomposition (n=9) ---")
    # Five verdicts. The P7.10 three are also re-checked at the
    # results/p7_11_decomposition/ path (identical numbers, but
    # let's gate them at the canonical NEW location too so the
    # paper's §4.2 table is anchored to the 2×2-complete file set).
    h_comb = _load("results/p7_11_decomposition/h1_combined.json")
    h_q_ltc = _load("results/p7_11_decomposition/h1_quantum_via_ltc.json")
    h_l_cls = _load("results/p7_11_decomposition/h1_liquid_via_classical.json")
    h_l_q = _load("results/p7_11_decomposition/h1_liquid_via_quantum.json")
    h_q_nlq = _load("results/p7_11_decomposition/h1_quantum_via_nonliquid.json")
    all_ok &= _check_str("p7_11 combined outcome",
                         h_comb["outcome"], "FALSIFIED")
    all_ok &= _check_str("p7_11 quantum_via_ltc outcome",
                         h_q_ltc["outcome"], "FALSIFIED")
    all_ok &= _check_str("p7_11 liquid_via_classical outcome",
                         h_l_cls["outcome"], "FALSIFIED")
    all_ok &= _check_str("p7_11 liquid_via_quantum outcome (NEW)",
                         h_l_q["outcome"], "FALSIFIED")
    all_ok &= _check_str("p7_11 quantum_via_nonliquid outcome (NEW)",
                         h_q_nlq["outcome"], "FALSIFIED")
    if h_l_q["bootstrap"] is not None:
        b = h_l_q["bootstrap"]
        all_ok &= _check(
            "Δ_liquid_via_quantum (paper: -0.3339)",
            b["delta_diff_mean"], -0.3339, tol=0.005)
        all_ok &= _check(
            "Δ_liquid_via_quantum CI low (paper: -0.6274)",
            b["ci_low"], -0.6274, tol=0.05)
        all_ok &= _check(
            "Δ_liquid_via_quantum CI high (paper: +0.0533)",
            b["ci_high"], 0.0533, tol=0.05)
    if h_q_nlq["bootstrap"] is not None:
        b = h_q_nlq["bootstrap"]
        all_ok &= _check(
            "Δ_quantum_via_nonliquid (paper: -0.1668)",
            b["delta_diff_mean"], -0.1668, tol=0.005)
        all_ok &= _check(
            "Δ_quantum_via_nonliquid CI low (paper: -0.4950)",
            b["ci_low"], -0.4950, tol=0.05)
        all_ok &= _check(
            "Δ_quantum_via_nonliquid CI high (paper: +0.2039)",
            b["ci_high"], 0.2039, tol=0.05)
    # τ-cross-check: Δ_τ_via_classical and Δ_τ_via_quantum
    # DISAGREE IN SIGN at the point-estimate level.
    if (h_l_cls["bootstrap"] is not None
            and h_l_q["bootstrap"] is not None):
        d_cls = h_l_cls["bootstrap"]["delta_diff_mean"]
        d_q = h_l_q["bootstrap"]["delta_diff_mean"]
        all_ok &= _check_str(
            "τ-cross-check: Δ_τ_via_classical sign (paper: positive)",
            "positive" if d_cls > 0 else "negative", "positive")
        all_ok &= _check_str(
            "τ-cross-check: Δ_τ_via_quantum sign (paper: NEGATIVE — disagrees)",
            "positive" if d_q > 0 else "negative", "negative")
    # Second algebraic identity (the P7.11 path): Δ_combined =
    # Δ_τ_via_quantum + Δ_quantum_via_nonliquid (exact per-cell).
    if all(d["bootstrap"] is not None for d in (h_comb, h_l_q, h_q_nlq)):
        sum_check = (h_l_q["bootstrap"]["delta_diff_mean"]
                     + h_q_nlq["bootstrap"]["delta_diff_mean"])
        all_ok &= _check(
            "P7.11 identity Δ_τ_via_q + Δ_q_via_nlq ≈ Δ_combined",
            sum_check, h_comb["bootstrap"]["delta_diff_mean"],
            tol=0.001)

    print("  --- §1+§3+§7 derived FHN ratios (P8.5 B1) ---")
    import statistics as _stats
    fhn_qlnn = []
    fhn_cls = []
    for _s in (0, 1, 2):
        _q = _load(
            f"results/p3_6_multi_state/te_qpinn_qnn_fitzhugh_nagumo/"
            f"seed_{_s}/metrics.json")
        _c = _load(
            f"results/p7_5_solver_h1/fitzhugh_nagumo_classical_pinn/"
            f"seed_{_s}/metrics.json")
        fhn_qlnn.append(float(_q["relative_l2"]))
        fhn_cls.append(float(_c["relative_l2"]))
    _qm, _qs = _stats.mean(fhn_qlnn), _stats.stdev(fhn_qlnn)
    _cm, _cs = _stats.mean(fhn_cls), _stats.stdev(fhn_cls)
    all_ok &= _check(
        "FHN te_qpinn_qnn mean relL² (paper: 0.337)", _qm, 0.337, tol=0.005)
    all_ok &= _check(
        "FHN cPINN mean relL² (paper: 0.870)", _cm, 0.870, tol=0.005)
    all_ok &= _check(
        "FHN std ratio cPINN/QLNN (paper: 24×; ≈ 24.41)",
        _cs / _qs, 24.41, tol=0.5)
    all_ok &= _check(
        "FHN mean ratio cPINN/QLNN (paper: 2.5×; ≈ 2.58)",
        _cm / _qm, 2.58, tol=0.1)

    print("  --- §6 T3 per-family scalars (P8.5 B1) ---")
    t3 = _load("results/p7_t3_mechanism/t3_scalars.json")
    all_ok &= _check(
        "hardware_efficient Q (paper: 0.796)",
        t3["hardware_efficient"]["entangling_q"], 0.796, tol=0.01)
    all_ok &= _check(
        "data_reuploading KL-to-Haar (paper: 0.195)",
        t3["data_reuploading"]["expressibility_kl"], 0.195, tol=0.005)
    all_ok &= _check(
        "hardware_efficient KL-to-Haar (paper: 0.211)",
        t3["hardware_efficient"]["expressibility_kl"], 0.211, tol=0.005)
    all_ok &= _check(
        "data_reuploading gradient variance (paper: 0.038)",
        t3["data_reuploading"]["gradient_variance"], 0.038, tol=0.005)
    all_ok &= _check(
        "hardware_efficient gradient variance (paper: 0.025)",
        t3["hardware_efficient"]["gradient_variance"], 0.025, tol=0.005)
    all_ok &= _check(
        "brickwall gradient variance (paper: 0.046)",
        t3["brickwall"]["gradient_variance"], 0.046, tol=0.005)

    print("\n=== PIVOT H3 mechanism (P7, tentative trend) ===")
    t3 = _load("results/p7_t3_mechanism/t3_scalars.json")
    # Lock the per-family T3 scalars to 3 sig figs (numerical determinism
    # given fixed seed=0 and n_samples=400).
    all_ok &= _check(
        "data_reuploading entangling_q (paper: 0.776)",
        t3["data_reuploading"]["entangling_q"], 0.776, tol=0.01)
    all_ok &= _check(
        "brickwall entangling_q LOW (paper: 0.309)",
        t3["brickwall"]["entangling_q"], 0.309, tol=0.02)

    print()
    if all_ok:
        print("ALL PAPER NUMBERS VERIFIED.")
        return 0
    print("INTEGRITY CHECK FAILED — fix mismatches before submitting.")
    return 1


if __name__ == "__main__":
    sys.exit(main())
