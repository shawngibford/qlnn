# results/p7_t3_mechanism/

P7 H3 mechanism: 4 T3 diagnostics per forecaster family at
the P4 sweep config (num_qubits=3, num_layers=1). Headline
output:

  - `t3_scalars.json`: per-family
      {expressibility_kl, entangling_q, gradient_variance,
       fourier_bandwidth}
  - `gradient_scaling.json`: BP scaling vs n_qubits per family

These scalars are cross-tabulated against P5's per-cell Δ
values (results/p5_h1_verdict/per_cell_records.json) by
scripts/make_p7_mechanism_figure.py to surface the property
that best predicts the inverted regime-dependent advantage.
