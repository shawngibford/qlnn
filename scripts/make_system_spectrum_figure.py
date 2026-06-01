"""Render the per-system Fourier-spectrum figure.

For each of the 4 P3.6 systems show the reference (numerical-integrator)
trajectory first-state component AND its Fourier amplitude spectrum
on log-y. The spectrum visualizes the K_max accessible to the quantum
PINN's data-reuploading Fourier series and where it cuts the signal
off — making the broadband-systems-are-hard argument visceral.

Reads:
  results/p3_6_multi_state/{family}_{system}/seed_0/curves.npz
    — uses the qcpinn family arbitrarily as a u_ref source
      (u_ref is the system's analytic / RK4 reference; identical
      across families and seeds for a given system).

Emits paper/figures/fig_system_spectrum.{png,pdf}.

Standalone — does NOT enter the integrity contract.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np

matplotlib.use("Agg")
plt.rcParams.update({
    "font.size": 10, "axes.titlesize": 11, "axes.labelsize": 9.5,
    "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8.5,
    "figure.dpi": 100, "savefig.dpi": 300,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.05,
})

REPO_ROOT = Path(__file__).resolve().parents[1]
IN_DIR = REPO_ROOT / "results" / "p3_6_multi_state"
OUT_DIR = REPO_ROOT / "paper" / "figures"

SYSTEMS = [
    ("lotka_volterra", "Lotka-Volterra  [S]",  "#0072B2"),
    ("van_der_pol",    "Van der Pol  [S]",     "#56B4E9"),
    ("fitzhugh_nagumo","FitzHugh-Nagumo  [B]", "#D55E00"),
    ("lorenz",         "Lorenz  [B]",          "#B71C1C"),
]
# Per the Schuld–Sweke–Meyer theorem (cited in §6): K_max = L · n.
# Our default chebyshev_dqc at n=3 qubits, L=1 → K_max = 3.
K_MAX_DEFAULT = 3


def _load_ref(system: str) -> tuple[np.ndarray, np.ndarray]:
    p = IN_DIR / f"qcpinn_{system}" / "seed_0" / "curves.npz"
    if not p.exists():
        p = IN_DIR / f"chebyshev_dqc_{system}" / "seed_0" / "curves.npz"
    z = np.load(p)
    return z["t_eval"], z["u_ref"]


def main() -> None:
    fig, axes = plt.subplots(
        2, 4, figsize=(14.0, 6.6),
        constrained_layout=True)

    for col, (system, label, color) in enumerate(SYSTEMS):
        t, u_ref = _load_ref(system)
        u0 = u_ref[:, 0]
        # ----- top row: time-domain trajectory -----
        ax_t = axes[0, col]
        ax_t.plot(t, u0, color=color, lw=1.6)
        ax_t.set_title(label, fontsize=10.5)
        ax_t.set_xlabel("t")
        ax_t.set_ylabel(r"$u_0(t)$")
        ax_t.grid(alpha=0.25, lw=0.5)

        # ----- bottom row: Fourier amplitude spectrum -----
        # x-axis is Fourier MODE ORDER k = ω·T/(2π); the
        # data-reuploading circuit at L=1, n=3 can represent integer
        # modes k = 0, 1, 2, 3. K_max=3 thus always sits at x=3
        # regardless of the system's time horizon, making the band
        # directly comparable across panels.
        ax_f = axes[1, col]
        N = len(u0)
        T = float(t[-1] - t[0])
        amps = np.abs(np.fft.rfft(u0 - u0.mean())) / N
        amps = np.clip(amps, 1e-8, None)
        # Fourier mode order k for each FFT bin
        k_order = np.arange(len(amps))
        ax_f.semilogy(k_order, amps, color=color, lw=1.4)
        # Mark the K_max cutoff line + grey-band region for k <= K_max
        ax_f.axvline(K_MAX_DEFAULT, color="#444444", lw=1.2, ls="--",
                      alpha=0.85)
        ax_f.fill_betweenx(
            [amps.min(), amps.max() * 1.3],
            0, K_MAX_DEFAULT, color="#888888", alpha=0.13)
        ax_f.text(K_MAX_DEFAULT + 0.3, amps.max() * 0.55,
                   rf"$K_{{\max}} = {K_MAX_DEFAULT}$",
                   fontsize=9, color="#444444",
                   ha="left", va="center")
        # Annotate how much spectral energy sits OUTSIDE K_max
        inside_mask = k_order <= K_MAX_DEFAULT
        e_total = float(np.sum(amps ** 2))
        e_outside = float(np.sum(amps[~inside_mask] ** 2))
        frac_outside = e_outside / max(e_total, 1e-12)
        ax_f.text(0.97, 0.97,
                   f"E outside $K_{{\\max}}$\n= {frac_outside:.1%}",
                   transform=ax_f.transAxes, ha="right", va="top",
                   fontsize=8.5,
                   bbox=dict(boxstyle="round,pad=0.3",
                             fc="lightyellow",
                             ec="goldenrod", lw=0.7))
        ax_f.set_xlabel(r"Fourier mode order  $k = \omega T / (2\pi)$")
        ax_f.set_ylabel("amplitude  (log)")
        ax_f.set_xlim(0, min(30, len(amps) - 1))
        ax_f.grid(alpha=0.25, lw=0.5, which="both")

    fig.suptitle(
        "Per-system reference dynamics  (top)  +  Fourier amplitude "
        "spectrum  (bottom)\n"
        r"grey band / vertical line = accessible Fourier bandwidth "
        r"at the program's $L\!=\!1, n\!=\!3$ default ($K_{\max}\!=\!3$)",
        y=1.06, fontsize=11)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "fig_system_spectrum.png")
    fig.savefig(OUT_DIR / "fig_system_spectrum.pdf")
    plt.close(fig)
    print(f"  wrote {OUT_DIR / 'fig_system_spectrum.pdf'}")


if __name__ == "__main__":
    main()
