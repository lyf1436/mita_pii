"""Task-independent trainer plumbing: config loading, model id, folder layout, dataset merging and splitting,
optuna parameter sampling, multi-device launch."""
import copy
import os
import subprocess
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:          # Python < 3.11
    import tomli as tomllib

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
WEIGHTS_ROOT = REPO_ROOT / "model_weights"
LOGS_ROOT = REPO_ROOT / "training_logs"
ID_KEYS = ("version", "dataset_nickname", "task_nickname", "model_name", "job_name")
SECTIONS = ("classification", "training", "optuna", "accelerate")


def load_config(path):
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    missing = [k for k in (*ID_KEYS, "dataset_paths", *SECTIONS) if k not in cfg]
    if missing:
        raise KeyError(f"{path} is missing {missing}")
    cfg["_path"] = str(Path(path).resolve())
    return cfg


def model_id(cfg):
    return "_".join(str(cfg[k]) for k in ID_KEYS)


def resolve(path):
    """Config paths are relative to the repository root."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_merged_dataset(paths):
    from datasets import concatenate_datasets, load_from_disk
    parts = [load_from_disk(str(resolve(p))) for p in paths]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def train_valid_split(ds, valid_fraction, valid_subsets=(), seed=0):
    """Whole `subset`s if `valid_subsets` is given (no near-duplicate frames shared with training),
    otherwise a seeded random `valid_fraction` of the rows."""
    if valid_subsets:
        subsets = ds["subset"]
        unknown = set(valid_subsets) - set(subsets)
        if unknown:
            raise ValueError(f"valid_subsets {sorted(unknown)} not in the datasets ({sorted(set(subsets))})")
        is_valid = np.isin(subsets, list(valid_subsets))
        return ds.select(np.flatnonzero(~is_valid)), ds.select(np.flatnonzero(is_valid))
    split = ds.train_test_split(test_size=valid_fraction, seed=seed, shuffle=True)
    return split["train"], split["test"]


def suggest_params(trial, cfg):
    """[optuna.params] entries -> a copy of cfg with those [training] keys replaced by sampled values.

    Each entry: {type = "float" | "int", low, high, log = false, step} or {type = "categorical", choices}.
    Every tuned name must be an existing [training] key, so a tuned run is an ordinary training run."""
    cfg = copy.deepcopy(cfg)
    for name, spec in cfg["optuna"].get("params", {}).items():
        if name not in cfg["training"]:
            raise KeyError(f"[optuna.params] {name!r} is not a [training] key")
        kind = spec["type"]
        if kind == "float":
            value = trial.suggest_float(name, spec["low"], spec["high"], log=spec.get("log", False), step=spec.get("step"))
        elif kind == "int":
            value = trial.suggest_int(name, spec["low"], spec["high"], log=spec.get("log", False), step=spec.get("step", 1))
        elif kind == "categorical":
            value = trial.suggest_categorical(name, spec["choices"])
        else:
            raise ValueError(f"[optuna.params] {name!r}: unknown type {kind!r}")
        cfg["training"][name] = value
    if "epochs" in cfg["optuna"]:                # shorter runs while searching
        cfg["training"]["epochs"] = cfg["optuna"]["epochs"]
    return cfg


def launched_by_accelerate():
    return "LOCAL_RANK" in os.environ or "ACCELERATE_LAUNCHED" in os.environ


def relaunch_with_accelerate(module, num_devices, mixed_precision):
    """Re-run `python -m module <same args>` under `accelerate launch` on `num_devices` processes."""
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(num_devices),
           "--mixed_precision", mixed_precision]
    if num_devices > 1:
        cmd.append("--multi_gpu")
    cmd += ["-m", module, *sys.argv[1:]]
    env = dict(os.environ, ACCELERATE_LAUNCHED="1")
    sys.exit(subprocess.call(cmd, cwd=REPO_ROOT, env=env))
