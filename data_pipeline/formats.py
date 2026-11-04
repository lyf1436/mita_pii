"""The formats prepared datasets are saved in.

Assumption: whatever its source, every prepared dataset is saved in one of these formats.

Expectation: two datasets in the same format can be merged as they are, and a label value means the same class
in both.

Rationale: the columns and the class list of a format are fixed in one place, so separately prepared sources
cannot drift apart. A class list only grows by appending, so the classes already in it keep their values and
datasets prepared earlier stay mergeable with newer ones."""
from datasets import Features, Image, Sequence, Value

# segmentation: `mask` is a single-channel PNG whose pixel values are indices into this list
SEGMENTATION_CLASSES = ["background", "face", "plate"]

SEGMENTATION_FEATURES = Features({
    "image_id": Value("string"),   # unique within a source
    "source": Value("string"),     # e.g. "pp4av"
    "subset": Value("string"),     # source-specific grouping (PP4AV: city); used for group-wise splits
    "image": Image(),
    "mask": Image(),
})

# detection: the same classes as segmentation, one box per object
DETECTION_CLASSES = SEGMENTATION_CLASSES

DETECTION_FEATURES = Features({
    "image_id": Value("string"),   # unique within a source
    "source": Value("string"),
    "subset": Value("string"),     # source-specific grouping; used for group-wise splits
    "image": Image(),
    "width": Value("int32"),       # of the image as decoded, orientation tag applied
    "height": Value("int32"),
    "objects": Sequence({
        "bbox": Sequence(Value("float32"), length=4),   # x, y, width, height in pixels of the decoded image
        "category": Value("int32"),                     # index into DETECTION_CLASSES
    }),
})

# unlabeled: images only, e.g. for inference demos; same columns as segmentation minus the mask
UNLABELED_FEATURES = Features({
    "image_id": Value("string"),   # path relative to the input folder
    "source": Value("string"),
    "subset": Value("string"),     # the image's folder, relative to the input folder ("" at the top level)
    "image": Image(),
})


def split_names(source, fmt):
    """Names of the train and test datasets that one source contributes in one format."""
    return f"{source}_{fmt}_train", f"{source}_{fmt}_test"
