# ACCESS Explore / Anvil materials

Use this for the advisor conversation and ACCESS Explore application.

## Ask

Authorize an ACCESS Explore request for Purdue Anvil so the remaining
audit-driven QLNN benchmark re-runs can run as an embarrassingly
parallel HPC job instead of on a laptop.

## Project abstract

> This project supports a pre-registered benchmark of Quantum Liquid
> Neural Networks against matched classical baselines on ordinary and
> partial differential equation solver/forecaster tasks. The current
> manuscript is complete in draft form (`paper/main.pdf`, 25 pages, plus
> an 8-page supplement), with 405 successful result cells, zero recorded
> training errors, 22 documented pre-registration amendments, and a
> mechanical integrity gate that verifies every cited number against
> committed JSON results. The pre-registered quantum-advantage
> hypothesis is currently falsified under matched controls; the remaining
> compute strengthens the submission by completing audit-driven fairness
> re-runs before PRX Quantum submission.
>
> The Anvil workload is approximately 225 independent training cells
> covering kuramoto/KdV completion, uniform-budget solver re-runs,
> qcpinn quantum-attribution variants, and forecaster re-runs after
> ansatz/budget fixes. The jobs are independent by `(system, model,
> seed)` cell and are therefore natural SLURM-array tasks. Estimated
> serial footprint is ~53 CPU-hours for committed scope, plus an optional
> ~40 CPU-hours for a PDE-side qcpinn attribution extension. Software
> stack: Python 3.11, JAX, Equinox, Diffrax, PennyLane, and GPU-capable
> PennyLane Lightning/cuQuantum where available.
>
> Target output: a strengthened, integrity-gated *PRX Quantum*
> submission and public reproducibility package.

## Advisor letter draft

> **[University Letterhead]**
>
> [Date]
>
> ACCESS Allocation Review Committee
>
> Re: ACCESS Explore allocation in support of [Student Name]'s doctoral
> research
>
> Dear Reviewers,
>
> I support the ACCESS Explore allocation request submitted by [Student
> Name]. The proposed project, a pre-registered benchmark of quantum
> neural differential-equation models against matched classical baselines,
> is part of [Student Name]'s dissertation research.
>
> The requested compute will complete an audit-driven re-run matrix of
> independent ODE/PDE training cells needed to strengthen a manuscript
> currently in preparation for *PRX Quantum*. The workload is
> embarrassingly parallel and well suited to Purdue Anvil's GPU resources.
>
> This work is separate from my other funded grants and does not duplicate
> compute resources available through those grants. I support the request
> and confirm [Student Name]'s qualifications to execute the work.
>
> Sincerely,
>
> [Advisor Name]  
> [Title, Department]  
> [University]  
> [Contact email]

## Checklist

- ACCESS account with institutional email.
- CV, <= 3 pages.
- Project abstract above, edited to final voice.
- Advisor letter on letterhead, signed.
- Resource target: Anvil GPU or Anvil AI.
- Requested credits: Explore-tier maximum unless advisor prefers a
  smaller ask.

## After award

1. Clone repo to `/scratch/$USER/qlnn`.
2. Install Python 3.11 environment.
3. Run one smoke cell.
4. Launch the SLURM array only after smoke passes.
5. Copy results back and run the paper integrity gate before any paper
   number changes.
