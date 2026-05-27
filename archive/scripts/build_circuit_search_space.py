"""Parametric circuit-search-space table — the paper's main-text
representation of the quantum search space (the standard QML way:
describe the axes × levels + a deduplicated topology list, rather than
print 28 individual diagrams in the body; the full diagram gallery is a
supplement, see scripts/make_diagnostic_figures.py fig_circuit_gallery_*).

Reads the qlnn configs in configs/unified_matrix/ (the single source of
truth for what actually runs), deduplicates by circuit topology
(family × qubits × layers × encoding × entanglement — REGIME is a
training knob, not topology, so it does not multiply circuits), and
emits:

  results/circuit_search_space/search_space_table.md   — paper table
  results/circuit_search_space/topologies.json         — machine-readable

Usage:
    python scripts/build_circuit_search_space.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CFG = REPO_ROOT / "configs" / "unified_matrix"
OUT = REPO_ROOT / "results" / "circuit_search_space"


def _topology_key(model: dict) -> tuple:
    a = model["ansatz"]
    p = a.get("params", {}) or {}
    ent = p.get("entanglement", "template")          # se/brickwall: fixed
    return (a["name"], int(model["num_qubits"]),
            int(model["num_layers"]), p.get("encoding", "rx"), ent)


def main() -> None:
    if not CFG.exists():
        raise SystemExit("run scripts/generate_unified_matrix.py first")

    topo: dict[tuple, list[str]] = defaultdict(list)
    axes = {"family": set(), "num_qubits": set(), "num_layers": set(),
            "encoding": set(), "entanglement": set()}
    for f in sorted(CFG.glob("qzeta_od__qlnn_*.yaml")):
        y = yaml.safe_load(f.read_text())
        m = y["model"]
        k = _topology_key(m)
        topo[k].append(y["unified_matrix"]["model"])
        axes["family"].add(k[0]); axes["num_qubits"].add(k[1])
        axes["num_layers"].add(k[2]); axes["encoding"].add(k[3])
        axes["entanglement"].add(k[4])

    OUT.mkdir(parents=True, exist_ok=True)
    rows = sorted(topo)

    # JSON
    payload = {
        "n_distinct_topologies": len(rows),
        "axes": {a: sorted(map(str, v)) for a, v in axes.items()},
        "topologies": [
            {"family": k[0], "num_qubits": k[1], "num_layers": k[2],
             "encoding": k[3], "entanglement": k[4],
             "model_keys": sorted(set(topo[k]))}
            for k in rows
        ],
    }
    (OUT / "topologies.json").write_text(json.dumps(payload, indent=2) + "\n")

    # Markdown
    md = ["# Quantum circuit search space (parametric)\n"]
    md.append("Distinct QLNN circuit **topologies** in the unified matrix. "
              "A regime (R0–R3) is a *training* hyperparameter — it does "
              "NOT change the circuit diagram — so the 16 family×regime "
              "cells collapse to 4 baseline topologies; the rest are the "
              "axis-ablation + dedup'd Optuna + promoted topologies.\n")
    md.append("## Search axes\n")
    md.append("| Axis | Levels |")
    md.append("|---|---|")
    md.append(f"| ansatz family | {', '.join(sorted(axes['family']))} |")
    md.append(f"| num_qubits | {', '.join(map(str, sorted(axes['num_qubits'])))} |")
    md.append(f"| num_layers | {', '.join(map(str, sorted(axes['num_layers'])))} |")
    md.append(f"| encoding | {', '.join(sorted(axes['encoding']))} |")
    md.append(f"| entanglement | {', '.join(sorted(axes['entanglement']))} "
              f"(`template` = ansatz controls it internally) |")
    md.append(f"\n**{len(rows)} distinct circuit topologies** "
              f"(supplement gallery renders each via qml.draw_mpl).\n")
    md.append("## Topology list\n")
    md.append("| # | Family | Qubits | Layers | Encoding | Entanglement |")
    md.append("|---|---|---|---|---|---|")
    for i, k in enumerate(rows, 1):
        md.append(f"| {i} | {k[0]} | {k[1]} | {k[2]} | {k[3]} | {k[4]} |")
    (OUT / "search_space_table.md").write_text("\n".join(md) + "\n")

    print(f"wrote {OUT}/search_space_table.md + topologies.json "
          f"({len(rows)} distinct topologies)")


if __name__ == "__main__":
    main()
