#!/usr/bin/env bash
# Build the supplement PDF.
#
# Usage:
#   bash paper/build_supplement.sh
#
# Output: paper/supplement.pdf.
set -euo pipefail
cd "$(dirname "$0")"

echo "[build_supplement] pdflatex pass 1/3"
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > /dev/null

echo "[build_supplement] bibtex"
bibtex supplement > /dev/null || {
    echo "[build_supplement] bibtex returned non-zero; continuing"
}

echo "[build_supplement] pdflatex pass 2/3"
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > /dev/null

echo "[build_supplement] pdflatex pass 3/3"
pdflatex -interaction=nonstopmode -halt-on-error supplement.tex > /dev/null

if [ ! -f supplement.pdf ]; then
    echo "[build_supplement] FAIL: supplement.pdf was not produced"
    exit 1
fi

PAGES=$(pdfinfo supplement.pdf | awk '/^Pages:/{print $2}')
SIZE=$(du -h supplement.pdf | awk '{print $1}')
echo "[build_supplement] DONE — supplement.pdf ($PAGES pages, $SIZE)"
