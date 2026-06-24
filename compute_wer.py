#!/usr/bin/env python3
"""Compute Word Error Rate (WER) between reference and two hypothesis files.

Usage examples (run on vast.ai terminal):
  python compute_wer.py --ref reference.txt --base base_output.txt --finetuned finetuned_output.txt

Each input file should contain one sample per line (aligned), or use directories.
Outputs a summary to stdout and a CSV `wer_per_sample.csv` if per-sample results are computed.
"""
import argparse
import sys
import os
from jiwer import wer, compute_measures
import csv


def read_lines(path):
    if os.path.isdir(path):
        raise ValueError(f"Expected file but got directory: {path}")
    with open(path, "r", encoding="utf-8") as f:
        lines = [l.strip() for l in f.readlines()]
    # filter out empty lines
    return [l for l in lines if l != ""]


def compute_overall(refs, hyps):
    # join with newline to compute corpus-level WER using jiwer
    joined_ref = "\n".join(refs)
    joined_hyp = "\n".join(hyps)
    m = compute_measures(joined_ref, joined_hyp)
    return m


def main():
    p = argparse.ArgumentParser(description="Compute WER between reference and two hypothesis files")
    p.add_argument("--ref", required=True, help="Reference file (one sample per line)")
    p.add_argument("--base", required=True, help="Base model hypothesis file (one sample per line)")
    p.add_argument("--finetuned", required=True, help="Fine-tuned model hypothesis file (one sample per line)")
    p.add_argument("--csv", default="wer_per_sample.csv", help="Output CSV path for per-sample WER")
    args = p.parse_args()

    refs = read_lines(args.ref)
    base = read_lines(args.base)
    finetuned = read_lines(args.finetuned)

    if not (len(refs) == len(base) == len(finetuned)):
        print(f"Warning: line counts differ: ref={len(refs)} base={len(base)} finetuned={len(finetuned)}", file=sys.stderr)

    n = min(len(refs), len(base), len(finetuned))
    refs = refs[:n]
    base = base[:n]
    finetuned = finetuned[:n]

    # corpus-level measures
    m_base = compute_overall(refs, base)
    m_finetuned = compute_overall(refs, finetuned)

    print("Corpus-level WER")
    print(f"Base model WER: {m_base['wer']:.4f} (sub={m_base['substitutions']} del={m_base['deletions']} ins={m_base['insertions']} ref_words={m_base['reference_words']})")
    print(f"Fine-tuned model WER: {m_finetuned['wer']:.4f} (sub={m_finetuned['substitutions']} del={m_finetuned['deletions']} ins={m_finetuned['insertions']} ref_words={m_finetuned['reference_words']})")
    try:
        delta = (m_base['wer'] - m_finetuned['wer'])
        print(f"Delta WER (base - finetuned): {delta:.4f}")
    except Exception:
        pass

    # per-sample WER and CSV
    rows = []
    total_base = 0.0
    total_finetuned = 0.0
    for i, (r, b, f) in enumerate(zip(refs, base, finetuned), start=1):
        try:
            w_base = wer(r, b)
            w_finetuned = wer(r, f)
        except Exception:
            w_base = float('nan')
            w_finetuned = float('nan')
        rows.append((i, r, b, f, w_base, w_finetuned))
        if not (w_base != w_base):
            total_base += w_base
        if not (w_finetuned != w_finetuned):
            total_finetuned += w_finetuned

    avg_base = total_base / len(rows) if rows else float('nan')
    avg_finetuned = total_finetuned / len(rows) if rows else float('nan')
    print("\nPer-sample average WER")
    print(f"Base avg WER: {avg_base:.4f}")
    print(f"Fine-tuned avg WER: {avg_finetuned:.4f}")

    # write CSV
    with open(args.csv, "w", encoding="utf-8", newline="") as csvf:
        writer = csv.writer(csvf)
        writer.writerow(["index", "reference", "base", "finetuned", "wer_base", "wer_finetuned"]) 
        for r in rows:
            writer.writerow(r)

    print(f"Per-sample WERs written to {args.csv}")


if __name__ == '__main__':
    main()
