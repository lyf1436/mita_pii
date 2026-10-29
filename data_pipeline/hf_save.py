"""Saving a dataset that is built row by row.

Assumption: the dataset can be too large to hold in memory, so it is built on disk before it is saved.

Expectation: afterwards only the saved dataset remains: no build copy is left in the home cache or next to the
output. If the build copy cannot be removed, the save still counts as done and the leftover folder is reported.

Rationale: building in a scratch folder next to the output keeps the large intermediate copy on the same disk as
the result, and out of the home folder. The built dataset is released before its scratch folder is deleted,
because some network filesystems refuse to delete files that are still open."""
import gc
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk


def save_generated(generator, features, gen_kwargs, out_dir, num_proc=1):
    """Build a dataset from the generator's rows, save it to the output folder, and return the saved dataset."""
    out_dir = Path(out_dir)
    out_dir.parent.mkdir(parents=True, exist_ok=True)
    cache = Path(tempfile.mkdtemp(prefix=f".{out_dir.name}_build_", dir=out_dir.parent))
    try:
        ds = Dataset.from_generator(generator, features=features, gen_kwargs=gen_kwargs,
                                    num_proc=num_proc if num_proc > 1 else None, cache_dir=str(cache))
        ds.save_to_disk(str(out_dir))
        # the dataset memory-maps its cache files; on network filesystems (e.g. NFS) a folder with open files
        # cannot be deleted ("Directory not empty"), so release it first
        del ds
    finally:
        gc.collect()
        shutil.rmtree(cache, ignore_errors=True)
        if cache.exists():
            print(f"warning: could not fully delete the build cache {cache}; it is safe to delete by hand")
    return load_from_disk(str(out_dir))
