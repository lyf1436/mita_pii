"""Search training settings for a segmentation model.

Usage:
    python -m trainer.segmentation_opt --config config/v0_0_1/segmentation.toml

Assumption: the config names the settings to search and their ranges, and one trial fits on one GPU.

Expectation: a search that can be stopped and continued, curves for every trial, and the best trial's model and
settings in the experiment's weights folder.

Rationale:
- A search only has to rank settings, not finish models. Trials therefore stop under a looser rule than final
  training, and a trial that falls behind the others at the same point is abandoned.
- Each trial uses one GPU. With several GPUs, trials run side by side: for many short runs that is simpler and
  faster than spreading each trial over all of them.
- The search is kept on disk, so running the command again continues it, and the configured number of trials
  counts trials from earlier runs too."""
import argparse
import json
import os
import subprocess
import sys

import optuna
from optuna.trial import TrialState

from trainer.common import LOGS_ROOT, REPO_ROOT, load_config, model_id, study_storage, suggest_params
from trainer.segmentation import default_weights_dir, save_model, train

PRUNERS = {"median": optuna.pruners.MedianPruner, "none": optuna.pruners.NopPruner}


def create_study(cfg):
    o = cfg["optuna"]
    (LOGS_ROOT / model_id(cfg)).mkdir(parents=True, exist_ok=True)
    return optuna.create_study(study_name=model_id(cfg), storage=study_storage(cfg), load_if_exists=True,
                               direction=o["direction"], pruner=PRUNERS[o.get("pruner", "median")]())


def objective(trial, cfg):
    o, mid = cfg["optuna"], model_id(cfg)
    trial_cfg = suggest_params(trial, cfg)
    trial_cfg["accelerate"] = dict(trial_cfg["accelerate"], num_devices=1)

    def report(metrics, epoch):
        trial.report(metrics[o["metric"]], epoch)
        if trial.should_prune():
            raise optuna.TrialPruned()

    metrics, model = train(trial_cfg, log_name=f"{mid}/trial_{trial.number:03d}", weights_dir=None, report=report)
    value = metrics[o["metric"]]

    done = trial.study.get_trials(deepcopy=False, states=(TrialState.COMPLETE,))
    better = (max if o["direction"] == "maximize" else min)([value] + [t.value for t in done]) == value
    if better:
        out = default_weights_dir(cfg)
        save_model(model, out, trial_cfg, dict(trial=trial.number, valid=metrics))
        (out / "best_params.json").write_text(json.dumps(dict(trial=trial.number, value=value, params=trial.params), indent=2))
        print(f"trial {trial.number}: new best {o['metric']} = {value:.4f}, saved to {out}")
    return value


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)   # set when spawned per GPU
    args = ap.parse_args()
    cfg = load_config(args.config)
    o, n_devices = cfg["optuna"], cfg["accelerate"]["num_devices"]
    study = create_study(cfg)

    if n_devices > 1 and not args.worker:
        procs = [subprocess.Popen([sys.executable, "-m", "trainer.segmentation_opt", "--config", args.config, "--worker"],
                                  cwd=REPO_ROOT, env=dict(os.environ, CUDA_VISIBLE_DEVICES=str(d)))
                 for d in range(n_devices)]
        codes = [p.wait() for p in procs]
        if any(codes):
            sys.exit(f"worker exit codes: {codes}")
    else:
        # n_trials counts across all workers and earlier runs of the same study
        counted = (TrialState.COMPLETE, TrialState.PRUNED)
        remaining = o["n_trials"] - len(study.get_trials(deepcopy=False, states=counted))
        if remaining > 0:
            stop = optuna.study.MaxTrialsCallback(o["n_trials"], states=counted)
            study.optimize(lambda trial: objective(trial, cfg), n_trials=remaining, callbacks=[stop])
        if args.worker:
            return

    best = study.best_trial
    print(f"best trial {best.number}: {o['metric']} = {best.value:.4f}  params = {best.params}")


if __name__ == "__main__":
    main()
