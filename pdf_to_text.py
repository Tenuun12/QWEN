#!/usr/bin/env python3
"""Extract text lines from a PDF and write to a plain text file.

Usage:
  python pdf_to_text.py input.pdf output.txt

This script uses PyMuPDF (fitz) to extract page-level text. Each non-empty
line in the PDF becomes a line in the output text file. Useful as a pre-step
before running `compute_wer.py` on two PDF documents.
"""
import sys
import os

try:
    import fitz  # PyMuPDF
except Exception as e:
    print("PyMuPDF (fitz) is required. Install via: pip install PyMuPDF", file=sys.stderr)
    raise


def extract_lines(pdf_path):
    doc = fitz.open(pdf_path)
    lines = []
    for page in doc:
        text = page.get_text("text")
        for ln in text.splitlines():
            ln = ln.strip()
            if ln:
                lines.append(ln)
    return lines


def main():
    if len(sys.argv) < 3:
        print("Usage: python pdf_to_text.py input.pdf output.txt", file=sys.stderr)
        sys.exit(2)
    inp = sys.argv[1]
    out = sys.argv[2]
    if not os.path.exists(inp):
        print(f"Input not found: {inp}", file=sys.stderr)
        sys.exit(1)
    lines = extract_lines(inp)
    with open(out, "w", encoding="utf-8") as f:
        for ln in lines:
            f.write(ln + "\n")
    print(f"Wrote {len(lines)} lines to {out}")


if __name__ == '__main__':
    main()
