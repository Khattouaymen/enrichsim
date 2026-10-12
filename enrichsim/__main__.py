"""Interface en ligne de commande.

Exemples :
    python -m enrichsim run main --n 20 --steps 20000
    python -m enrichsim run all  --n 20
    python -m enrichsim report results/main
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from .experiment import DESCRIPTIONS, DESIGNS, run_experiment


def _progress(done: int, total: int, group: str) -> None:
    width = 30
    filled = int(width * done / total)
    sys.stdout.write(f"\r  [{'#' * filled}{'.' * (width - filled)}] {done}/{total}  ({group})      ")
    sys.stdout.flush()
    if done == total:
        sys.stdout.write("\n")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="enrichsim", description="Simulation de l'enrichissement environnemental")
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("run", help="lancer une expérience")
    r.add_argument("design", choices=[*DESIGNS, "all"])
    r.add_argument("--n", type=int, default=20, help="sujets par groupe (défaut 20)")
    r.add_argument("--steps", type=int, default=20_000, help="durée de l'élevage (défaut 20000)")
    r.add_argument("--workers", type=int, default=None, help="processus parallèles")
    r.add_argument("--seed", type=int, default=2026)
    r.add_argument("--out", default="results")
    r.add_argument("--no-report", action="store_true")

    rep = sub.add_parser("report", help="analyser un dossier de résultats")
    rep.add_argument("folder")

    args = ap.parse_args(argv)
    if args.cmd == "run":
        designs = list(DESIGNS) if args.design == "all" else [args.design]
        for d in designs:
            print(f"\n=== {d} : {DESCRIPTIONS[d]} ===")
            t = time.time()
            folder = run_experiment(d, n_subjects=args.n, steps=args.steps,
                                    out_dir=Path(args.out) / d, workers=args.workers,
                                    seed=args.seed, progress=_progress)
            print(f"  données : {folder}  ({time.time() - t:.0f} s)")
            if not args.no_report:
                from .report import build_report
                print(f"  rapport : {build_report(folder)}")
    elif args.cmd == "report":
        from .report import build_report
        print(build_report(Path(args.folder)))


if __name__ == "__main__":
    main()
