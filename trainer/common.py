"""The task-independent parts of training: reading a config, naming its results, merging and splitting its data,
sampling tuned values, and spreading one command over several devices.

Assumption: a config describes one experiment completely, and its identifying fields differ between experiments.

Expectation: the same config always gives the same model id, the same results folders and the same
train/validation split.

Rationale:
- The model id is built from the config's identifying fields, so any weights or logs folder can be traced back to
  the experiment that made it.
- Paths inside a config are read relative to the repository, so a config means the same thing wherever a command
  is started.
- Every tuned value must replace an existing training setting. A tuned run is then an ordinary run with other
  values, and the best values can be reused unchanged for a final run.
- Search-only settings apply while tuning only, so a search can cut runs shorter than a final run does, without
  the final run inheriting that."""
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
    """Paths inside a config are relative to the repository."""
    path = Path(path)
    return path if path.is_absolute() else REPO_ROOT / path


def load_merged_dataset(paths):
    from datasets import concatenate_datasets, load_from_disk
    parts = [load_from_disk(str(resolve(p))) for p in paths]
    return parts[0] if len(parts) == 1 else concatenate_datasets(parts)


def train_valid_split(ds, valid_fraction, valid_subsets=(), seed=0):
    """Validation is the named groups if any are given, so near-duplicate frames never end up on both sides;
    otherwise a random share of the rows, the same on every run."""
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
    """A copy of the config with each tuned setting replaced by a value sampled for this trial, and the
    search-only settings applied.

    Each range is {type = "float" | "int", low, high, log = false, step} or {type = "categorical", choices}."""
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
    return search_config(cfg)


def search_config(cfg):
    """A copy of the config with the search-only settings applied."""
    cfg = copy.deepcopy(cfg)
    for name, value in cfg["optuna"].get("training_overrides", {}).items():
        if name not in cfg["training"]:
            raise KeyError(f"[optuna.training_overrides] {name!r} is not a [training] key")
        cfg["training"][name] = value
    return cfg


def study_storage(cfg):
    return cfg["optuna"].get("storage") or f"sqlite:///{LOGS_ROOT / model_id(cfg) / 'optuna.db'}"


def launched_by_accelerate():
    return "LOCAL_RANK" in os.environ or "ACCELERATE_LAUNCHED" in os.environ


def relaunch_with_accelerate(module, num_devices, mixed_precision):
    """Start this same command again as one process per device, wait for it, and exit with its result."""
    cmd = [sys.executable, "-m", "accelerate.commands.launch", "--num_processes", str(num_devices),
           "--mixed_precision", mixed_precision]
    if num_devices > 1:
        cmd.append("--multi_gpu")
    cmd += ["-m", module, *sys.argv[1:]]
    env = dict(os.environ, ACCELERATE_LAUNCHED="1")
    sys.exit(subprocess.call(cmd, cwd=REPO_ROOT, env=env))
