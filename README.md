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
are indices into `SEGMENTATION_CLASSES` (`background`, `face`, `plate`). Unlabeled: the same columns without `mask`,
one dataset `<source>_unlabeled` per folder, for inference demos only.

### Config sections

- top level: `version`, `dataset_paths` (merged by the trainer), `dataset_nickname`, `task_nickname`, `model_name`, `job_name`
- `[classification]`: the classes the model outputs
- `[training]`: everything the trainer reads (model, data split, augmentation, optimisation, logging, early stopping).
  Logging and evaluation are given as counts over the planned run (`num_logs`, `num_evals`), not step intervals, so
  they fall at the same share of training images for any batch size, and early stopping compares runs fairly
- `[optuna]`: search settings; `[optuna.params]`, where each entry tunes the `[training]` key of the same name;
  `[optuna.training_overrides]`, `[training]` values used only while searching (e.g. a larger early-stopping threshold)
- `[accelerate]`: `num_devices`, `mixed_precision`

## Workflow

An example run with the default config, `config/v1/segmentation.toml` (model id
`v1_pp4av_seg3_mita_segmentation_vit_tiny`).

Run every command from the repository root. The scripts import each other's folders (`data_pipeline`, `models`,
`trainer`) as packages, and `python -m` finds them in the current folder. Paths inside a config are resolved from
the repository, but command-line paths (`--config`, `--save_dir`, `--input_dir`, `--raw_dir`) are relative to the
current folder.

### 0. Setup (once)

```bash
pip install -r requirements.txt
```

MiTA-Segmenter's `segm` package must also be importable. Install it from its repository; this repo does not include it.

### 1. Build the datasets

```bash
python -m data_pipeline.pp4av_to_segmentation --save_dir data/v1 --raw_dir /data/pp4av
```

- `--raw_dir` is where PP4AV is downloaded and extracted (default `data/raw/pp4av`, about 4.5 GB). Point it at an
  existing download (the notebooks used `/data/pp4av`) to skip downloading again.
- Writes `data/v1/pp4av_segmentation_train` (every city except the test ones) and `data/v1/pp4av_segmentation_test`
  (switzerland and strasbourg, set by `--test_subsets`), and prints the image count and cities of each.
- Stops if the folders exist; add `--overwrite` to rebuild them.

Optional, unlabeled images for the inference demo:

```bash
python -m data_pipeline.folder_to_unlabeled --input_dir /path/to/photos --save_dir data/v1
```

Every image under the folder and its subfolders goes into `data/v1/<folder name>_unlabeled` (`--source` sets the
name). Files that cannot be read as images are skipped and listed.

### 2. Edit the config

| key | what to decide |
|---|---|
| `job_name` | changes the model id, so different experiments don't overwrite each other |
| `[training] backbone` | e.g. `vit_small_patch16_384` |
| `[training] epochs` | the longest a run may last; early stopping can end it sooner |
| `[training] valid_subsets` | whole cities for validation, so it shares no near-duplicate frames with training; empty = a random `valid_fraction` |
| `[optuna.params]` | what to tune, and the ranges |
| `[accelerate] num_devices` | number of GPUs |

### 3. Calibrate the early-stopping thresholds (recommended, once)

The default thresholds are guesses. Check them against one real run:

```bash
python -m trainer.segmentation_acc --config config/v1/segmentation.toml
tensorboard --logdir training_logs
```

Look at `valid/loss_slope` (how fast the validation loss falls per evaluation) and `valid/mean_dice` (when it stops
improving). Then set:

- `[training] early_stopping_min_decrease`: roughly the slope where `mean_dice` flattens out
- `[optuna.training_overrides] early_stopping_min_decrease`: a few times larger, so tuning trials stop sooner

You can stop this run with Ctrl-C once you've seen enough; step 5 overwrites its weights.

### 4. Tune

```bash
python -m trainer.segmentation_opt --config config/v1/segmentation.toml
```

- Runs `n_trials` trials. Each samples `[optuna.params]`, applies `[optuna.training_overrides]` (the larger
  threshold), and trains on one GPU. With `num_devices = N`, N trials run in parallel, one per GPU.
- Optuna stops a trial early when its `metric` falls behind the others at the same evaluation.
- Watch it: `tensorboard --logdir training_logs/<model id>` shows each trial as `trial_000`, `trial_001`, ...
- The study is saved in `training_logs/<model id>/optuna.db`. The best trial's weights and `best_params.json` go to
  `model_weights/<model id>/`.
- Running the same command again continues the study. To search more, raise `n_trials` and run it again.

### 5. Train the final model from scratch

```bash
python -m trainer.segmentation_acc --config config/v1/segmentation.toml --best_params
```

- Uses the best trial's tuned values; everything else, including the smaller early-stopping threshold, comes from
  `[training]`. It prints which trial it took.
- Trains on `num_devices` GPUs, relaunching itself under `accelerate launch` when there is more than one.
- Replaces `model_weights/<model id>/` with this run:
  - `config.json`, `model.safetensors`: the model
  - `config.toml`: a copy of the config
  - `train_state.json`: step, images seen, final validation scores, the trial used (`_best_trial`), and why it
    stopped (`stopped_early`, empty if it ran all epochs)
- Its TensorBoard curves are at the top level of `training_logs/<model id>/`, next to the trial folders.

### 6. Evaluate and use the model

Open `segmentation_inference.ipynb`, set `WEIGHTS_DIR = "model_weights/<model id>"`, `TEST_PATHS`, and optionally
`UNLABELED_PATHS` (e.g. `["data/v1/photos_unlabeled"]`), then run it top to bottom. It shows:

1. test Dice and IoU next to the validation scores from training. A large gap means validation was too easy (e.g.
   near-duplicate frames); set `valid_subsets` if so
2. the test images with the most faces and plates: input, P(private), blurred result
3. close-ups of individual faces and plates, before and after blurring
4. random unlabeled images: prediction and blurred result
5. any image file: set `IMAGE_PATH` (and `OUTPUT_PATH` to save the result)

### Later experiments

- **Another setting on the same data:** change `job_name` and repeat steps 4–6.
- **New data:** write `data_pipeline/<source>_to_segmentation.py` and run it with `--save_dir data/v2`, add
  `config/v2/segmentation.toml` with every train split to merge in `dataset_paths`, and repeat steps 3–6.

## Adding things

- **a source**: `data_pipeline/<source>_to_<format>.py` writes `<save_dir>/<source>_<format>_{train,test}` with the
  format's features. List the new train split in the config's `dataset_paths`.
- **a task** (e.g. bounding boxes): add its format to `formats.py`, a model in `models/`, `trainer/<task>{,_acc,_opt}.py`
  and `config/<version>/<task>.toml`. `trainer/common.py` (config, model id, merging, splitting, optuna sampling,
  multi-device launch) is task-independent.
