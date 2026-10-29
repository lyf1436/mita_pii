"""Output formats shared by every data_pipeline script.

A format fixes the columns and the label vocabulary, so datasets from different sources saved in the same
format can be merged with `datasets.concatenate_datasets`. When a new source brings a new class, append it to
the vocabulary: existing ids keep their meaning and older datasets stay mergeable.
"""
from datasets import Features, Image, Value

# segmentation: `mask` is a single-channel PNG whose pixel values are indices into this list
SEGMENTATION_CLASSES = ["background", "face", "plate"]

SEGMENTATION_FEATURES = Features({
    "image_id": Value("string"),   # unique within a source
    "source": Value("string"),     # e.g. "pp4av"
    "subset": Value("string"),     # source-specific grouping (PP4AV: city); used for group-wise splits
    "image": Image(),
    "mask": Image(),
})

# unlabeled: images only, e.g. for inference demos; same columns as segmentation minus the mask
UNLABELED_FEATURES = Features({
    "image_id": Value("string"),   # path relative to the input folder
    "source": Value("string"),
    "subset": Value("string"),     # the image's folder, relative to the input folder ("" at the top level)
    "image": Image(),
})


def split_names(source, fmt):
    """The two dataset directories a source contributes to a data version: <source>_<format>_{train,test}."""
    return f"{source}_{fmt}_train", f"{source}_{fmt}_test"
