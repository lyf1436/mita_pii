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
| Each data command is self-contained, and its data names its own labels | No shared code fixes the classes or their order. Every segmentation command saves the same columns, including `label_names` (the class name of each mask value), so the datasets concatenate; the trainer takes the union of the names as the model's classes, and the same name in two sources is one class. |
| Tuning and final training are separate steps | Tuning tries many settings with short runs that stop early. The final model is trained from scratch with the best settings and a stricter stopping rule. |
| Progress is counted in training images, not steps | Logging, validation and stopping fall at the same points of a run whatever the batch size or number of GPUs, so such runs are compared fairly. |

## Folders

| folder | what you find there |
|---|---|
| `data_pipeline/` | commands that prepare a data source, each on its own |
| `data/<version>/` | prepared datasets: `<source>_<format>_train` and `<source>_<format>_test` |
| `config/<version>/` | one experiment config per task |
| `model_weights/<model id>/` | the trained model, a copy of its config, and its final scores |
| `training_logs/<model id>/` | TensorBoard curves, and the tuning search with one folder per trial |
| `models/`, `trainer/` | the model and the training commands |
| `inference/` | the command that blurs images with a trained model |

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

Optional: the internal four-task photo set (COCO boxes). A box's label is its category's supercategory;
`--label_rename` renames them (several to one name merges them, `null` drops them). To keep only the licence plates,
as the class `plate` that PP4AV also has, rename `license_plate` and drop every other supercategory (the command
prints the labels it ends up with, and names the file's supercategories if a key is not one of them):

```bash
python -m data_pipeline.coco_to_segmentation --json data/raw/train_4task.json --save_dir data/v0_0_1 --source geico \
    --label_rename '{"license_plate": "plate", "<other supercategory>": null}' --num_proc 16
```

- You get `data/v0_0_1/geico_segmentation_train` and `data/v0_0_1/geico_segmentation_test`; a tenth of the photo
  folders is held out for testing (`--test_fraction`).
- Images that cannot be read, or whose size does not fit their annotations, are listed and left out.
- Categories named `NOT_...` are left out (`--negative_prefix`). Images with no kept box get an all-background mask.
- `--num_proc` worker processes read the images in parallel.
- Add the two folders to `dataset_paths` in the config to train on them alongside PP4AV. The model's classes are the
  union of the label names of all listed datasets: background first, then the others alphabetically.

The same photo set as boxes instead of masks, for measuring object sizes in step 2:

```bash
python -m data_pipeline.coco_to_detection --json data/raw/train_4task.json --save_dir data/v0_0_1 --source geico \
    --label_rename '{"license_plate": "plate", "<other supercategory>": null}'
```

You get `data/v0_0_1/geico_detection_train` and `data/v0_0_1/geico_detection_test`, each box with its label's name.
With the same file and `--test_fraction`, they hold the same images on the same side of the split as the segmentation
version.

Optional: your own images, to try the model on later.

```bash
python -m data_pipeline.folder_to_unlabeled --input_dir /path/to/photos --save_dir data/v0_0_1
```

Every image in the folder and its subfolders ends up in `data/v0_0_1/photos_unlabeled` (named after the folder; set
your own name with `--source`). Files that are not readable images are listed and left out.

### 2. Set up the experiment

First choose the training scale range. Open `detection_eda.ipynb` from the repository root, set `DATASET_PATH` to a
detection dataset (e.g. `data/v0_0_1/geico_detection_train`) and run it. It shows how large faces and plates are, how
large they become at inference, and how a candidate range spreads them between one token (too small to find) and the
model input (too large to see whole).

Then open `config/v0_0_1/segmentation.toml` and decide:

| setting | what to decide |
|---|---|
| `job_name` | a name for this experiment, so it doesn't overwrite another one |
| `backbone` | model size, e.g. `vit_small_patch16_384` |
| `scale_range` | the random resize factors of training crops, from `detection_eda.ipynb`. Validation and inference always resize the whole image so its longer side fits the model input. |
| `rotation_range`, `corner_jitter` | the random rotation (degrees) and corner shifts (share of width and height) applied to each training image before it is rescaled and cropped. The largest upright rectangle inside the warped image is kept, so no warp border enters training. |
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

You can press Ctrl-C once you have seen enough. Tuning in step 4 replaces this model.

### 4. Tune

```bash
python -m trainer.segmentation_opt --config config/v0_0_1/segmentation.toml
```

- Runs the configured number of trials, each with different settings. Weak trials are abandoned early.
- Watch them in TensorBoard: each trial appears as `trial_000`, `trial_001`, ...
- The best trial's model and settings are saved to `model_weights/<model id>/`.
- If interrupted, run the same command again to continue. To search longer, raise `n_trials` and run it again.

### 5. Train the final model

1. Write a new config for the final run: a copy of the tuning config with a new `job_name` (e.g. `tiny_final`),
   so the final model gets its own weights and logs folders.
2. Put the best trial's settings into its `[training]` section. They are listed in `best_params.json` in the
   tuning experiment's `model_weights/<model id>/` folder.
3. Train it:

```bash
python -m trainer.segmentation_acc --config config/v0_0_1/segmentation_final.toml
```

- Trains from scratch under the final stopping rule. The config is the only source of settings, so the copy saved
  next to the model shows exactly what trained it.
- The model is saved to `model_weights/<final model id>/`. `train_state.json` there records its scores and why
  it stopped.

### 6. Check and use the model

Open `segmentation_inference.ipynb`. Set the weights folder to the final model's, the test data and, optionally, your unlabeled data,
then run it top to bottom. You get:

1. test scores next to the validation scores from training. If test is much lower, validation was too easy: name
   validation cities in step 2 and train again.
2. for each label, a few random test images containing it: the input with its boxes, the blurred region, and the
   anonymized image
3. for each label, a grid of close-ups of 100 random objects, each original next to its anonymized crop

Sections 3, 4 and 5 draw new random samples every time their cells are run.
4. random images of your own, before and after blurring
5. any single image file, blurred and optionally saved

To blur a file or a whole folder of images without the notebook:

```bash
python -m inference.anonymize --weights model_weights/<model id> --input /path/to/photos --output_dir /path/to/out
```

Every readable image is resized whole to fit the model input, labelled in one pass, and written blurred under the same
relative path. `--batch_size` sets how many images share a forward pass (default 32; images of any sizes can share
one), `--save_masks` also writes the blurred region of each image, and `--threshold`, `--dilate_px` and `--blur_sigma`
set how much is blurred. `--dilate_px` and `--blur_sigma` are measured on an image whose longer side is
`--reference_size` (480) pixels and scale with each image, so a photo is blurred the same way at any resolution.

### Next experiments

- **Other settings, same data:** change `job_name`, then repeat steps 4 to 6.
- **More data:** prepare the new source into a new version folder such as `data/v0_0_2`, add a config under
  `config/v0_0_2/` that lists every dataset to train on, then repeat steps 3 to 6.
