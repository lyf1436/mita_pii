# mita_pii: blur faces and license plates with MiTA-Segmenter

Train a model that finds faces and license plates in street images, then use it to blur them.

`pp4av_eda.ipynb` and `pp4av_train.ipynb` are the prototypes that proved the approach works.

## What to expect

- **Assumption:** your images resemble street or dashcam footage, and faces and plates in them are often tiny,
  smaller than 16 pixels.
- **Expectation:** a trained model, a measure of how well it does on cities it never saw, and blurred images.
- **Every result can be traced back:** one config file describes one experiment. Its weights and logs are saved
  in folders named after it, so you can always tell which experiment produced a result.

## How the project is shaped, and why

| decision | why |
|---|---|
| Data is prepared once per source, into versioned folders | Training only ever reads prepared data, so adding a source never changes training. A new version leaves the data of earlier experiments untouched. |
| Every source gives a train part and a test part | Test images are held out from the start, so the test score always comes from images training never saw. |
| PP4AV's test part is whole cities | Images from one city are frames of one video and many are near-duplicates. A random split would test on images almost identical to training ones. |
| Prepared data of the same kind can be merged | A config lists the datasets to train on, so combining sources means editing a list. |
| Tuning and final training are separate steps | Tuning tries many settings with short runs that stop early. The final model is trained from scratch with the best settings and a stricter stopping rule. |
| Progress is counted in training images, not steps | Logging, validation and stopping fall at the same points of a run whatever the batch size or number of GPUs, so such runs are compared fairly. |

## Folders

| folder | what you find there |
|---|---|
| `data_pipeline/` | commands that prepare a data source |
| `data/<version>/` | prepared datasets: `<source>_<format>_train` and `<source>_<format>_test` |
| `config/<version>/` | one experiment config per task |
| `model_weights/<model id>/` | the trained model, a copy of its config, and its final scores |
| `training_logs/<model id>/` | TensorBoard curves, and the tuning search with one folder per trial |
| `models/`, `trainer/` | the model and the training commands |

The model id combines the config's `version`, `dataset_nickname`, `task_nickname`, `model_name` and `job_name`,
e.g. `v0_0_1_pp4av_seg3_mita_segmentation_tiny_test`.

`data/`, `model_weights/` and `training_logs/` are not tracked by git.

## Workflow

Run every command from the repository root.

### 0. Setup (once)

```bash
pip install -r requirements.txt
```

Also install MiTA-Segmenter from its own repository. This repo does not include it.

### 1. Prepare the data

```bash
python -m data_pipeline.pp4av_to_segmentation --save_dir data/v0_0_1 --raw_dir /data/pp4av
```

- `--raw_dir` is where PP4AV is downloaded to, about 4.5 GB. If you already downloaded it, point it there to reuse the download.
- You get `data/v0_0_1/pp4av_segmentation_train` and `data/v0_0_1/pp4av_segmentation_test`. The command prints how many
  images and which cities went into each.
- If the folders already exist the command stops. Add `--overwrite` to rebuild them.

Optional: your own images, to try the model on later.

```bash
python -m data_pipeline.folder_to_unlabeled --input_dir /path/to/photos --save_dir data/v0_0_1
```

Every image in the folder and its subfolders ends up in `data/v0_0_1/photos_unlabeled` (named after the folder; set
your own name with `--source`). Files that are not readable images are listed and left out.

### 2. Set up the experiment

Open `config/v0_0_1/segmentation.toml` and decide:

| setting | what to decide |
|---|---|
| `job_name` | a name for this experiment, so it doesn't overwrite another one |
| `backbone` | model size, e.g. `vit_small_patch16_384` |
| `epochs` | the longest a run may last; runs usually stop earlier |
| `valid_subsets` | cities to validate on. Leave empty to validate on a random tenth of the training images, which scores higher than real use. |
| `[optuna.params]` | which settings to tune, and over what range |
| `num_devices` | how many GPUs to use |

### 3. Calibrate when runs stop (once)

The stopping thresholds in the config are starting guesses. Train once and look at the curves:

```bash
python -m trainer.segmentation_acc --config config/v0_0_1/segmentation.toml
tensorboard --logdir training_logs
```

- Find where `valid/mean_dice` stops improving, and read `valid/loss_slope` there.
- Set the stopping threshold for final training (`early_stopping_min_decrease` in `[training]`) to roughly that
  value.
- Set the one for tuning (in `[optuna.training_overrides]`) a few times larger, so tuning runs stop sooner.

You can press Ctrl-C once you have seen enough. The final run in step 5 replaces this model.

### 4. Tune

```bash
python -m trainer.segmentation_opt --config config/v0_0_1/segmentation.toml
```

- Runs the configured number of trials, each with different settings. Weak trials are abandoned early.
- Watch them in TensorBoard: each trial appears as `trial_000`, `trial_001`, ...
- The best trial's model and settings are saved to `model_weights/<model id>/`.
- If interrupted, run the same command again to continue. To search longer, raise `n_trials` and run it again.

### 5. Train the final model

```bash
python -m trainer.segmentation_acc --config config/v0_0_1/segmentation.toml --best_params
```

- Trains from scratch with the best settings found in step 4, under the final stopping rule. It prints which
  trial's settings it used.
- The model in `model_weights/<model id>/` is replaced by this one. `train_state.json` there records its scores,
  the trial its settings came from, and why it stopped.

### 6. Check and use the model

Open `segmentation_inference.ipynb`. Set the weights folder, the test data and, optionally, your unlabeled data,
then run it top to bottom. You get:

1. test scores next to the validation scores from training. If test is much lower, validation was too easy: name
   validation cities in step 2 and train again.
2. the test images with the most faces and plates, before and after blurring
3. close-ups of single faces and plates, before and after blurring
4. random images of your own, before and after blurring
5. any single image file, blurred and optionally saved

### Next experiments

- **Other settings, same data:** change `job_name`, then repeat steps 4 to 6.
- **More data:** prepare the new source into a new version folder such as `data/v0_0_2`, add a config under
  `config/v0_0_2/` that lists every dataset to train on, then repeat steps 3 to 6.
