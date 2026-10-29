"""Train a segmentation model with accelerate on [accelerate] num_devices devices.

    python -m trainer.segmentation_acc --config config/v1/segmentation.toml

With num_devices > 1 the script relaunches itself under `accelerate launch`. Weights go to
model_weights/<model id>, TensorBoard logs to training_logs/<model id>.

--best_params replaces the tuned [training] keys with the best trial of this config's optuna study
(trainer.segmentation_opt); everything else, including early stopping, comes from [training].
"""
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
