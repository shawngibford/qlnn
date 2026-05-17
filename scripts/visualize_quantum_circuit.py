from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pennylane as qml
from sklearn.preprocessing import MinMaxScaler


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CSV = REPO_ROOT / "data" / "raw" / "qZETA_data_copy.csv"


def _load_qzeta(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    df = df.rename(columns={c: c.strip() for c in df.columns})
    if "TEMP EXT" in df.columns and "TEMP_EXT" not in df.columns:
        df = df.rename(columns={"TEMP EXT": "TEMP_EXT"})

    if "DATE" in df.columns:
        dt = pd.to_datetime(df["DATE"], format="mixed", dayfirst=True, errors="raise")
        df = df.assign(DATE=dt).sort_values("DATE").reset_index(drop=True)

    return df


def _make_angles_from_dataframe(
    df: pd.DataFrame,
    *,
    feature_cols: list[str],
    angle_scale: float,
) -> np.ndarray:
    if angle_scale <= 0:
        raise ValueError("angle_scale must be > 0")

    for c in feature_cols:
        if c not in df.columns:
            raise ValueError(f"Missing column in CSV: {c}")

    X = df[feature_cols].to_numpy(dtype=np.float64)
    scaler = MinMaxScaler()
    Xn = scaler.fit_transform(X)

    # Map [0, 1] -> [0, angle_scale]
    return Xn * angle_scale


def _entangle(*, n_qubits: int, pattern: str) -> None:
    if pattern == "linear":
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[w, w + 1])
    elif pattern == "ring":
        for w in range(n_qubits - 1):
            qml.CNOT(wires=[w, w + 1])
        if n_qubits > 2:
            qml.CNOT(wires=[n_qubits - 1, 0])
    elif pattern == "all_to_all":
        for i in range(n_qubits):
            for j in range(i + 1, n_qubits):
                qml.CNOT(wires=[i, j])
    else:
        raise ValueError(f"Unknown entanglement pattern: {pattern}")


def build_encoder_qnode(
    *,
    n_qubits: int,
    n_layers: int,
    reuploads: int,
    entanglement: str,
    device_name: str,
) -> qml.QNode:
    if n_qubits <= 0:
        raise ValueError("n_qubits must be positive")
    if n_layers <= 0:
        raise ValueError("n_layers must be positive")
    if reuploads <= 0:
        raise ValueError("reuploads must be positive")

    dev = qml.device(device_name, wires=n_qubits)

    @qml.qnode(dev)
    def circuit(x: np.ndarray, weights: np.ndarray):
        # x is expected shape: (reuploads * n_qubits,)
        if x.ndim != 1:
            raise ValueError("x must be 1D")
        if x.shape[0] != reuploads * n_qubits:
            raise ValueError(
                f"x must have length reuploads*n_qubits={reuploads*n_qubits}, got {x.shape[0]}"
            )
        if weights.shape != (n_layers, n_qubits, 2):
            raise ValueError(
                f"weights must have shape (n_layers, n_qubits, 2)={(n_layers, n_qubits, 2)}, got {weights.shape}"
            )

        idx = 0
        for _r in range(reuploads):
            # Angle encoding
            for w in range(n_qubits):
                qml.RY(x[idx + w], wires=w)
            idx += n_qubits

            for l in range(n_layers):
                # Trainable rotations
                for w in range(n_qubits):
                    qml.RX(weights[l, w, 0], wires=w)
                    qml.RZ(weights[l, w, 1], wires=w)

                _entangle(n_qubits=n_qubits, pattern=entanglement)

        return [qml.expval(qml.PauliZ(w)) for w in range(n_qubits)]

    return circuit


def _safe_specs_dict(spec: qml.resource.CircuitSpecs) -> dict:
    d = spec.to_dict()
    # Shots is not JSON serializable in newer PennyLane; stringify.
    d["shots"] = str(d.get("shots"))
    return d


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualize a simple PennyLane quantum feature-encoder circuit (ASCII + image) and optional dataset output distributions."
    )
    parser.add_argument("--n-qubits", type=int, default=6)
    parser.add_argument("--n-layers", type=int, default=2)
    parser.add_argument("--reuploads", type=int, default=1)
    parser.add_argument("--entanglement", choices=["linear", "ring", "all_to_all"], default="ring")
    parser.add_argument("--device", type=str, default="default.qubit")

    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV)
    parser.add_argument(
        "--feature-cols",
        nargs="+",
        default=["PRE", "TEMP_EXT", "TEMP_CULTURE", "PAR_LIGHT", "PH", "DO"],
        help="Feature columns used to generate input angles. Must match reuploads*n_qubits unless --tile-features is set.",
    )
    parser.add_argument("--tile-features", action="store_true", help="If set, tile/truncate features to match reuploads*n_qubits.")

    parser.add_argument("--angle-scale", type=float, default=float(2 * np.pi), help="Scale for angle encoding.")
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "results" / "quantum_circuit_viz",
    )

    parser.add_argument("--row-index", type=int, default=0, help="Row used for the example circuit draw.")
    parser.add_argument("--plot-output-distribution", action="store_true")
    parser.add_argument("--n-samples", type=int, default=100)

    args = parser.parse_args()

    required = int(args.n_qubits * args.reuploads)

    df = _load_qzeta(args.csv)
    angles = _make_angles_from_dataframe(df, feature_cols=list(args.feature_cols), angle_scale=float(args.angle_scale))

    if angles.shape[1] != required:
        if not args.tile_features:
            raise ValueError(
                f"feature-cols produced {angles.shape[1]} features, but reuploads*n_qubits={required}. "
                f"Either change --n-qubits/--reuploads or pass --tile-features."
            )
        # tile/truncate
        tiled = np.tile(angles, reps=(1, int(np.ceil(required / angles.shape[1]))))[:, :required]
        angles = tiled

    if not (0 <= args.row_index < len(df)):
        raise ValueError(f"row-index out of range: {args.row_index}")

    rng = np.random.default_rng(args.seed)
    weights = rng.normal(loc=0.0, scale=0.1, size=(args.n_layers, args.n_qubits, 2)).astype(np.float64)

    qnode = build_encoder_qnode(
        n_qubits=args.n_qubits,
        n_layers=args.n_layers,
        reuploads=args.reuploads,
        entanglement=args.entanglement,
        device_name=args.device,
    )

    x0 = angles[args.row_index].astype(np.float64)

    # ASCII draw
    print("\n=== PennyLane circuit (ASCII) ===")
    print(qml.draw(qnode)(x0, weights))

    # Specs/resources
    spec = qml.specs(qnode)(x0, weights)
    spec_dict = _safe_specs_dict(spec)

    out_dir: Path = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    (out_dir / "specs.json").write_text(json.dumps(spec_dict, indent=2) + "\n")

    # Matplotlib draw
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, _ax = qml.draw_mpl(qnode)(x0, weights)
    fig.suptitle(
        f"Quantum encoder: {args.n_qubits} qubits, {args.n_layers} layers, reuploads={args.reuploads}, ent={args.entanglement}",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_dir / "circuit.png", dpi=200)
    plt.close(fig)

    if args.plot_output_distribution:
        n_samples = int(args.n_samples)
        if n_samples <= 0:
            raise ValueError("n-samples must be positive")

        idx = rng.choice(len(df), size=min(n_samples, len(df)), replace=False)
        outs = []
        for i in idx:
            outs.append(np.asarray(qnode(angles[i].astype(np.float64), weights), dtype=np.float64))
        outs_arr = np.stack(outs, axis=0)  # (N, n_qubits)

        # Save summary stats
        stats = {
            "mean": outs_arr.mean(axis=0).tolist(),
            "std": outs_arr.std(axis=0).tolist(),
            "min": outs_arr.min(axis=0).tolist(),
            "max": outs_arr.max(axis=0).tolist(),
        }
        (out_dir / "output_stats.json").write_text(json.dumps(stats, indent=2) + "\n")

        # Plot histograms per wire
        fig, axes = plt.subplots(args.n_qubits, 1, figsize=(8, 2.0 * args.n_qubits), sharex=True)
        if args.n_qubits == 1:
            axes = [axes]

        for w in range(args.n_qubits):
            ax = axes[w]
            ax.hist(outs_arr[:, w], bins=40, alpha=0.8)
            ax.set_ylabel(f"wire {w}")
            ax.axvline(0.0, color="black", linewidth=1)

        axes[-1].set_xlabel("expval(PauliZ)")
        fig.suptitle("Quantum encoder outputs over random dataset samples", fontsize=10)
        fig.tight_layout()
        fig.savefig(out_dir / "output_distribution.png", dpi=200)
        plt.close(fig)

    print(f"\nSaved circuit artifacts to: {out_dir}")


if __name__ == "__main__":
    main()
