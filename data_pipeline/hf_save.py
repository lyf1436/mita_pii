"""Build a Hugging Face dataset from a generator and save it, without leaving a copy in ~/.cache."""
import gc
import shutil
import tempfile
from pathlib import Path

from datasets import Dataset, load_from_disk


def save_generated(generator, features, gen_kwargs, out_dir, num_proc=1):
    """Run `generator(**gen_kwargs)` into a scratch cache next to `out_dir`, save the result to `out_dir`,
    delete the cache, and return the saved dataset (loaded from `out_dir`)."""
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
