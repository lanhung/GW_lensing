#!/usr/bin/env bash
# Builds supplement.pdf: the text and figures (revtex) followed by the landscape pages of Table S3.
# Run in a clean copy of this directory (pdflatex picks up stale files in the output directory).
set -e
for f in supplement table_persystem_doc; do pdflatex -interaction=nonstopmode $f.tex > $f.build.log; pdflatex -interaction=nonstopmode $f.tex > $f.build.log; done
mv supplement.pdf supplement_text.pdf
pdfunite supplement_text.pdf table_persystem_doc.pdf supplement.pdf
