# mita_pii: blur faces and license plates with MiTA-Segmenter

`pp4av_eda.ipynb` and `pp4av_train.ipynb` are the prototypes that proved the approach. The code below is the same
pipeline split into reusable parts.

## Layout

| folder | holds | naming |
|---|---|---|
| `data_pipeline/` | one script per (source, format): fetch, preprocess, save as a Hugging Face dataset | `<source>_to_<format>.py` |
| `data/` | version dirs; each holds mergeable datasets, two per source | `data/<version>/<source>_<format>_{train,test}` |
| `models/` | one script per model | `mita_segmentation.py`, later `mita_bounding_box.py`, ... |
| `trainer/` | one trainer per task, each with an optuna and an accelerate variant | `<task>.py`, `<task>_opt.py`, `<task>_acc.py` |
| `config/` | version dirs, one TOML per task | `config/<version>/<task>.toml` |
| `model_weights/` | one flat folder per model id | `config.json`, `model.safetensors`, `config.toml`, `train_state.json` |
| `training_logs/` | one flat folder per model id (TensorBoard; optuna study and per-trial logs) | |

model id = `<version>_<dataset_nickname>_<task_nickname>_<model_name>_<job_name>` from the config.

`data/`, `model_weights/` and `training_logs/` are git-ignored.

### Formats

`data_pipeline/formats.py` defines each format's columns and label ids, so datasets in the same format merge with
`concatenate_datasets`. Segmentation: `image_id`, `source`, `subset`, `image`, and `mask`, a PNG whose pixel values
are indices into `SEGMENTATION_CLASSES` (`background`, `face`, `plate`).

### Config sections

- top level: `version`, `dataset_paths` (merged by the trainer), `dataset_nickname`, `task_nickname`, `model_name`, `job_name`
- `[classification]`: the classes the model outputs
- `[training]`: everything the trainer reads (model, data split, augmentation, optimisation, logging, early stopping).
  Logging and evaluation are given as counts over the planned run (`num_logs`, `num_evals`), not step intervals, so
  they fall at the same share of training images for any batch size, and early stopping compares runs fairly
- `[optuna]`: search settings; `[optuna.params]`, where each entry tunes the `[training]` key of the same name;
  `[optuna.training_overrides]`, `[training]` values used only while searching (e.g. a larger early-stopping threshold)
- `[accelerate]`: `num_devices`, `mixed_precision`

## Usage

MiTA-Segmenter's `segm` package must be importable. Install it from its repository; this repo does not include it.

```bash
pip install -r requirements.txt

# 1. data: writes data/v1/pp4av_segmentation_train and data/v1/pp4av_segmentation_test
python -m data_pipeline.pp4av_to_segmentation --save_dir data/v1

# 2a. train (relaunches itself under `accelerate launch` when [accelerate] num_devices > 1)
python -m trainer.segmentation_acc --config config/v1/segmentation.toml

# 2b. or tune first (one worker per device, sharing a study stored in training_logs/<model id>/optuna.db),
#     then train from scratch with the best trial's parameters
python -m trainer.segmentation_opt --config config/v1/segmentation.toml
python -m trainer.segmentation_acc --config config/v1/segmentation.toml --best_params

tensorboard --logdir training_logs
```

3. Inference: `segmentation_inference.ipynb` loads `model_weights/<model id>` and reports test metrics. It also shows
   examples and anonymizes any image file.

Run everything from the repository root. Config paths are relative to it.

## Adding things

- **a source**: `data_pipeline/<source>_to_<format>.py` writes `<save_dir>/<source>_<format>_{train,test}` with the
  format's features. List the new train split in the config's `dataset_paths`.
- **a task** (e.g. bounding boxes): add its format to `formats.py`, a model in `models/`, `trainer/<task>{,_acc,_opt}.py`
  and `config/<version>/<task>.toml`. `trainer/common.py` (config, model id, merging, splitting, optuna sampling,
  multi-device launch) is task-independent.
