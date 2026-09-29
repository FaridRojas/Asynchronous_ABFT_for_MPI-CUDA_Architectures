#!/usr/bin/env python3
"""Parse the GFLOPS table from Fault-Tolerant-SGEMM-on-NVIDIA-GPUs/sgemm's stdout.

Their stdout (the performance section) looks like:
    Matrix Size         |    1024|    2048|    3072|    4096|
    cublas              |   XXXX |   XXXX |   XXXX |   XXXX |
    kernel_sgemm_small  | ...
    abft_kernel_small   | ...

Converted to long-format CSV: kernel,size,gflops
"""
import argparse, csv, sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_txt")
    ap.add_argument("output_csv")
    args = ap.parse_args()

    sizes = None
    rows = []
    with open(args.input_txt) as f:
        for raw in f:
            line = raw.rstrip("\n")
            if "|" not in line:
                continue
            parts = [p.strip() for p in line.split("|")]
            # drop leading/trailing empties produced by leading/trailing '|'
            while parts and parts[0] == "":
                parts.pop(0)
            while parts and parts[-1] == "":
                parts.pop()
            if not parts:
                continue
            name = parts[0]
            if name == "Matrix Size":
                try:
                    sizes = [int(p) for p in parts[1:]]
                except ValueError:
                    pass
                continue
            if sizes is None:
                continue
            try:
                vals = [float(p) for p in parts[1:]]
            except ValueError:
                continue
            if len(vals) != len(sizes):
                continue
            rows.append((name, vals))

    if sizes is None:
        print("ERROR: no 'Matrix Size' header line found in", args.input_txt,
              file=sys.stderr)
        sys.exit(1)

    with open(args.output_csv, "w", newline="") as g:
        w = csv.writer(g)
        w.writerow(["kernel", "size", "gflops"])
        for name, vals in rows:
            for sz, gf in zip(sizes, vals):
                w.writerow([name, sz, gf])

    print(f"Wrote {args.output_csv} ({len(rows)} kernels x {len(sizes)} sizes)")


if __name__ == "__main__":
    main()
