"""Train a segmentation model from a config, on one GPU or several.

Usage:
    python -m trainer.segmentation_acc --config config/v0_0_1/segmentation.toml
    python -m trainer.segmentation_acc --config config/v0_0_1/segmentation.toml --best_params

Assumption: the config's datasets exist. With --best_params, the experiment has already been searched.

Expectation: the trained model in the experiment's weights folder and its curves in the experiment's logs folder.
With several GPUs configured, the same command trains one run across all of them.

Rationale: --best_params takes only the tuned values from the search. Everything else, including the stricter
stopping rule, comes from the config, so the final model is trained from scratch under its own rules."""
import argparse
import os

from trainer.common import (apply_best_params, launched_by_accelerate, load_config, model_id,
                            relaunch_with_accelerate)
from trainer.segmentation import default_weights_dir, train


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--best_params", action="store_true", help="use the best optuna trial's parameters")
    args = ap.parse_args()
    cfg = load_config(args.config)

    acc = cfg["accelerate"]
    if acc["num_devices"] > 1 and not launched_by_accelerate():
        relaunch_with_accelerate("trainer.segmentation_acc", acc["num_devices"], acc.get("mixed_precision", "no"))

    if args.best_params:
        cfg, best = apply_best_params(cfg)
        if int(os.environ.get("RANK", 0)) == 0:
            print(f"using optuna trial {best.number} ({cfg['optuna']['metric']} = {best.value:.4f}): {best.params}")
    metrics, _ = train(cfg, log_name=model_id(cfg), weights_dir=default_weights_dir(cfg))
    if int(os.environ.get("RANK", 0)) == 0:
        print(f"{model_id(cfg)}: {metrics}")


if __name__ == "__main__":
    main()
