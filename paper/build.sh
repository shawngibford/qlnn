#!/usr/bin/env bash
# Build the PRX Quantum paper PDF.
#
# Usage:
#   bash paper/build.sh
#
# Runs the standard 4-pass pdflatex + bibtex cycle that resolves all
# cross-references and citations. Output: paper/main.pdf.
#
# Exit non-zero if any pass fails or the final PDF doesn't exist.

set -euo pipefail
cd "$(dirname "$0")"

echo "[build_paper] pdflatex pass 1/4"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "[build_paper] bibtex"
bibtex main > /dev/null || {
    echo "[build_paper] bibtex returned non-zero (no citations yet?); continuing"
}

echo "[build_paper] pdflatex pass 2/4"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "[build_paper] pdflatex pass 3/4"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

echo "[build_paper] pdflatex pass 4/4"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > /dev/null

if [ ! -f main.pdf ]; then
    echo "[build_paper] FAIL: main.pdf was not produced"
    exit 1
fi

PAGES=$(pdfinfo main.pdf | awk '/^Pages:/{print $2}')
SIZE=$(du -h main.pdf | awk '{print $1}')
echo "[build_paper] DONE — main.pdf ($PAGES pages, $SIZE)"
