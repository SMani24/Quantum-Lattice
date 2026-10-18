#!/usr/bin/env python3
"""Score frozen MLP validation predictions with explicit D5 conventions."""
import os
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import argparse
import json
from pathlib import Path
import sys

from quantum_lattice.metrics.hamiltonian_evaluation import evaluate_campaigns

ROOT = Path(__file__).resolve().parents[2]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config",type=Path,default=ROOT/"configs/hamiltonian_scores.json")
    parser.add_argument("--output",type=Path,help="new directory inside the ignored Hamiltonian-score root")
    args=parser.parse_args(argv)
    config=json.loads(args.config.read_text())
    output,summary=evaluate_campaigns(ROOT,config,output=args.output)
    print("\nReference-sign D5; raw two-channel/full convention; development validation.")
    for row in summary["contrasts"]:
        if row["context"]=="matched_pipeline" and row["metric"]==summary["primary_metric"]:
            reduction=row['relative_reduction']
            relative=f"{100*reduction:.3f}%" if reduction is not None else "undefined"
            print(f"  {row['campaign']:10} L={row['L']}: f→f {row['f_mean_d5']:.6f}; "
                  f"g→g {row['g_mean_d5']:.6f}; reduction {relative}")
    print(f"Elapsed: {summary['elapsed_seconds']:.2f}s; results: {output/'RESULTS.md'}")
    return 0


if __name__=="__main__":
    try:
        raise SystemExit(main())
    except (ValueError,TypeError,KeyError,OSError) as error:
        print(f"Hamiltonian scoring stopped: {error}",file=sys.stderr)
        raise SystemExit(1)
