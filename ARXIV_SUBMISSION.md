# arXiv submission package

Stage-ready, NOT submitted. Submit only after advisor approval per
NEXT_STEPS.md Phase E timing.

---

## Submission classification

| Field | Value |
|---|---|
| **Primary archive** | `quant-ph` (Quantum Physics) |
| **Cross-list 1** | `cs.LG` (Machine Learning) |
| **Cross-list 2** | `physics.comp-ph` (Computational Physics) |
| **Submission type** | Research article |
| **Title** | Regime-dependent advantage in quantum physics-informed neural networks: a pre-registered comparison on an ODE/PDE hardness ladder |
| **Author** | Shawn Gibford (single author) |
| **arXiv submission abstract** | See plain-text variant below (<= 1920-char arXiv limit) |
| **License** | arXiv non-exclusive license; CC-BY-4.0 recommended |
| **Source bundle** | Files listed in §"Source tarball contents" |

## Plain-text abstract (arXiv submission form)

The arXiv submission form has a 1920-character single-paragraph
plain-text abstract field that strips LaTeX macros. The version
below substitutes the `\ddiff`, `\dsmooth`, `\dbroad`, `\relltwo`
macros with their rendered text and drops `\textemdash{}` glyphs.

```text
We pre-register and execute a head-to-head benchmark of four
quantum physics-informed neural network (PINN) families against
capacity-matched classical PINN and Neural-ODE baselines on a
hardness ladder of four ordinary differential equations
(Lotka-Volterra, Van der Pol, Lorenz, FitzHugh-Nagumo) and four
partial differential equations (heat, smooth Burgers, Allen-Cahn,
shock Burgers), each at three random seeds. Following the
methodology proposed by Bowles, Ahmed, and Schuld for QML
benchmarking, every comparison uses matched training budgets,
paired-bootstrap confidence intervals, an underfit guard, and a
known-structure skyline. The pre-registered hypothesis -- that
quantum PINNs achieve a regime-dependent advantage favoring
smooth/periodic systems over broadband/multiscale ones -- is
FALSIFIED across two PRIMARY verdicts (full-ladder solver task at
n=24, with the point-estimate sign inverted relative to the
original hypothesis; and forecaster task at n=9 with the CI
excluding zero negatively) and three additional layered
sensitivity points. The expanded matrix surfaces a positive
sub-finding: the trainable-embedding quantum PINN with QNN encoder
(te_qpinn_qnn) exhibits a 24x tighter seed standard deviation and
2.5x lower mean error than the classical PINN on FitzHugh-Nagumo,
a regime-dependent structural advantage on stiff fast-slow
dynamics. We release the full open-source benchmark framework,
including a verify_paper_integrity.py script that gates every
cited number against a committed JSON file, and document two
mechanism-proven-but-compute-deferred extensions (12-dimensional
Kuramoto, KdV with third-order spatial autodiff) for follow-up.
```

(Length: ~1900 chars — verify under arXiv's 1920 limit before submit.)

---

## Source tarball contents

arXiv expects a `.tar.gz` of LaTeX source + figures (no build
artifacts). The exact set:

```text
paper/main.tex
paper/sections/01_introduction.tex
paper/sections/02_methods.tex
paper/sections/03_solver_results.tex
paper/sections/04_forecaster_results.tex
paper/sections/05_h1_verdict.tex
paper/sections/06_mechanism.tex
paper/sections/07_discussion.tex
paper/sections/08_conclusions.tex
paper/references.bib
paper/main.bbl                  (arXiv prefers .bbl over re-running bibtex)
paper/supplement.tex            (separate arXiv entry — see §Supplement)
paper/supplement.bbl
paper/figures/fig_*.pdf         (25 files; PDFs only, no PNGs needed)
```

**Excluded** (build artifacts; arXiv compiles from source):
`paper/main.pdf`, `paper/supplement.pdf`, `paper/main_with_supplement.pdf`,
`paper/*.aux`, `paper/*.log`, `paper/*.out`, `paper/*.fls`,
`paper/*.fdb_latexmk`, `paper/*.synctex.gz`, `paper/*.toc`,
`paper/mainNotes.bib`, `paper/supplementNotes.bib`.

### Build command

```bash
# From repo root, after verifying tree is clean:
tar czf qlnn-arxiv-v1.tar.gz \
    paper/main.tex \
    paper/sections/*.tex \
    paper/references.bib \
    paper/main.bbl \
    paper/supplement.tex \
    paper/supplement.bbl \
    paper/figures/*.pdf

# Verify the tarball:
tar tzf qlnn-arxiv-v1.tar.gz | sort | head -20
tar tzf qlnn-arxiv-v1.tar.gz | wc -l   # should be ~35 files
```

### arXiv-side compilation test (no upload)

arXiv compiles every submission on its own server, so the
source must build cleanly with a stock `texlive` install. Local
test:

```bash
mkdir -p /tmp/arxiv-test && cd /tmp/arxiv-test
tar xzf ~/dev/phd/qlnn/qlnn-arxiv-v1.tar.gz
cd paper
pdflatex -interaction=nonstopmode main.tex
# Expected: main.pdf, 28 pages, matches local build.
```

---

## Supplement handling

arXiv treats the supplement as a separate "ancillary file" or as a
second tex root in the same source bundle. The latter is cleaner:
`paper/supplement.tex` references its own `.bbl` and uses
`\externaldocument{main}` (via `xr` package) to resolve `\ref{}`
calls into the main document. As long as `paper/main.aux` is in
the same directory at compile time, both compile to two separate
PDFs from one source tar.

Verify on local test:

```bash
cd /tmp/arxiv-test/paper
pdflatex -interaction=nonstopmode main.tex
pdflatex -interaction=nonstopmode supplement.tex
ls *.pdf   # main.pdf (28pp) + supplement.pdf (8pp)
```

---

## Tagging plan

Once the advisor approves submission:

```bash
# Pre-submission tag (release the source the arXiv version was
# generated from, for reproducibility):
git tag -a v0.3-arxiv-preprint -m "arXiv preprint v1 source"
git push origin v0.3-arxiv-preprint

# Build the tar from the tagged tree:
git archive --format=tar.gz \
    --prefix=qlnn-arxiv-v1/ \
    v0.3-arxiv-preprint -- \
    paper/main.tex paper/sections/*.tex paper/references.bib \
    paper/main.bbl paper/supplement.tex paper/supplement.bbl \
    paper/figures/*.pdf \
    > qlnn-arxiv-v1.tar.gz

# Then upload qlnn-arxiv-v1.tar.gz via the arXiv web form.
```

### v2 / v3 / etc.

- **Bump v2 when**: the post-Phase-C verdict refresh changes a
  cited number, the bibliography expands, or a reviewer comment
  before peer review prompts a substantive revision. Tag
  `v0.4-arxiv-preprint-v2` and re-upload via the arXiv "Replace"
  flow.
- **New paper instead of v2 when**: the underlying research
  question changes (a follow-up paper is a NEW arXiv submission,
  not a v-bump).

---

## Pre-upload checklist

Before pressing submit on the arXiv form:

- [ ] `bash paper/build.sh && bash paper/build_supplement.sh` exits 0
- [ ] `PYTHONPATH=src python scripts/verify_paper_integrity.py` exits 0
- [ ] Plain-text abstract is under 1920 chars
- [ ] Tarball contents match the file list above (no .aux / .log / .pdf)
- [ ] Local tarball-compile test passes (28pp main + 8pp supp)
- [ ] Author affiliation is correct on the submission form (currently
      "Independent" in `paper/main.tex`; if advisor relationship
      changes the affiliation, update here too)
- [ ] License choice confirmed (CC-BY-4.0 recommended)
- [ ] Crosslist categories (`cs.LG`, `physics.comp-ph`) checked
- [ ] Pre-registration repository linked in body (already there:
      `https://github.com/shawngibford/qlnn`)
- [ ] Tag `v0.3-arxiv-preprint` pushed to GitHub before upload
- [ ] Tarball generated FROM the tagged commit, not from working tree
