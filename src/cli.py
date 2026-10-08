"""Command-line entry point.

    python -m src.cli experiments
    python -m src.cli backfill --name architectures --horizons 5 20 --models lightgbm mlp
    python -m src.cli import-results results/architectures --name architectures

`backfill` runs a walk-forward experiment on the Sharadar research universe and
records it, with its protocol and git commit, in the research store.
`import-results` loads predictions already written to parquet by a script, so
long runs need not be repeated to land in the store.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path

import pandas as pd

from src.config import setup_logging

warnings.filterwarnings("ignore", message="X does not have valid feature names")


def cmd_experiments(args) -> None:
    from src import store

    df = store.experiments()
    if df.empty:
        print("The research store is empty.")
        return
    cols = ["id", "name", "model", "horizon", "target", "universe",
            "ic_mean", "ic_tstat_folds", "git_commit", "created_at"]
    pd.set_option("display.width", 220)
    print(df[[c for c in cols if c in df]].round(4).to_string(index=False))


def cmd_backfill(args) -> None:
    from src import experiment, store

    protocol = {"refit_every": args.refit_every, "train_stride": args.train_stride}
    panel = experiment.load_sharadar_panel(
        horizons=args.horizons, start=args.start, size=args.size,
        universe_mode=args.universe, seq_len=args.seq_len,
    )
    for h in args.horizons:
        for model in args.models:
            preds, sc = experiment.run(panel, horizon=h, model_names=[model],
                                       target_mode=args.target, **protocol)
            if sc.empty:
                print(f"{model} h={h}: no predictions")
                continue
            metrics = sc.iloc[0].drop(["model", "horizon", "target"]).to_dict()
            store.save_experiment(
                name=args.name, model=model, horizon=h, target=args.target,
                universe=args.universe,
                protocol={**protocol, "start": args.start, "size": args.size, "seq_len": args.seq_len},
                predictions=preds, metrics=metrics,
            )
            print(f"{model} h={h}: IC {metrics.get('ic_mean', float('nan')):.4f} — stored")


def cmd_import(args) -> None:
    """Load predictions_h{h}[_{model}].parquet files written by the scripts."""
    from src import store
    from src.config import MIN_NAMES_PER_DATE
    from src.metrics import evaluate

    folder = Path(args.folder)
    files = sorted(folder.glob("predictions_h*.parquet"))
    if not files:
        print(f"No predictions_h*.parquet files in {folder}")
        return

    for path in files:
        horizon = int(re.match(r"predictions_h(\d+)", path.stem).group(1))
        preds = pd.read_parquet(path)
        for model, group in preds.groupby("model"):
            target = group["target"].iloc[0] if "target" in group else "cross_sectional"
            metrics = evaluate(group, horizon=horizon, min_names=MIN_NAMES_PER_DATE,
                               refit_every=args.refit_every)
            store.save_experiment(
                name=args.name, model=model, horizon=horizon, target=target,
                universe=args.universe,
                protocol={"refit_every": args.refit_every, "train_stride": args.train_stride,
                          "imported_from": str(path)},
                predictions=group, metrics=metrics,
            )
            print(f"{path.name}: {model} — IC {metrics.get('ic_mean', float('nan')):.4f} imported")


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="python -m src.cli")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("experiments", help="list recorded experiments")

    bf = sub.add_parser("backfill", help="run and store a walk-forward experiment")
    bf.add_argument("--name", required=True)
    bf.add_argument("--horizons", type=int, nargs="+", default=[5, 20])
    bf.add_argument("--models", nargs="+", default=["lightgbm"])
    bf.add_argument("--target", default="cross_sectional", choices=["cross_sectional", "absolute"])
    bf.add_argument("--universe", default="point_in_time",
                    choices=["point_in_time", "survivors", "todays_list"])
    bf.add_argument("--start", default="2005-01-01")
    bf.add_argument("--size", type=int, default=1000)
    bf.add_argument("--seq-len", type=int, default=0)
    bf.add_argument("--refit-every", type=int, default=63)
    bf.add_argument("--train-stride", type=int, default=5)

    im = sub.add_parser("import-results", help="load script-written predictions into the store")
    im.add_argument("folder")
    im.add_argument("--name", required=True)
    im.add_argument("--universe", default="point_in_time")
    im.add_argument("--refit-every", type=int, default=63)
    im.add_argument("--train-stride", type=int, default=5)

    args = parser.parse_args(argv)
    setup_logging("WARNING")
    {"experiments": cmd_experiments, "backfill": cmd_backfill,
     "import-results": cmd_import}[args.command](args)


if __name__ == "__main__":
    main(sys.argv[1:])
