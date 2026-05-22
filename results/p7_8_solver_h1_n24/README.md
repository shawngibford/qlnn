# results/p7_8_solver_h1_n24/

P7.8 commit — combined ODE+PDE solver-task H1 verdict at
**n=24** (12 smooth + 12 broad), 1.33× the P7.6 n=18 verdict.

Adds two pre-reg §4 systems:
  - fitzhugh_nagumo (ODE, BROADBAND/MULTISCALE)
  - burgers_shock   (PDE, BROADBAND/MULTISCALE)

Skips (deferred to follow-up paper, see PRE_REG_AMENDMENT A11):
  - kuramoto (12D high-dim, ~7 hr/cell)
  - kdv (jacrev³ mechanism gate PASS but integrated cost ~8 hr/seed)
