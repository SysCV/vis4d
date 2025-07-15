"""Script to convert a dataset to hdf5 format."""

from __future__ import annotations

import argparse
import os

import numpy as np
from tqdm import tqdm

from vis4d.common.imports import H5PY_AVAILABLE

if H5PY_AVAILABLE:
    import h5py
else:
    raise ImportError("Please install h5py to enable HDF5Backend.")


def convert_dataset(source_dir: str) -> None:
    """Convert a dataset to HDF5 format.

    This function converts an arbitary dictionary to an HDF5 file. The keys
    inside the HDF5 file preserve the directory structure of the original.

    As an example, if you convert "/path/to/dataset" to HDF5, the resulting
    file will be: "/path/to/dataset.hdf5". The file "relative/path/to/file"
    will be stored at "relative/path/to/file" inside /path/to/dataset.hdf5.

    Args:
        source_dir (str): The path to the dataset to convert.
    """
    if not os.path.exists(source_dir):
        raise FileNotFoundError(f"No such file or directory: {source_dir}")

    source_dir = os.path.join(source_dir, "")  # must end with trailing slash
    hdf5_path = source_dir.rstrip("/") + ".hdf5"
    if os.path.exists(hdf5_path):
        print(f"File {hdf5_path} already exists! Skipping {source_dir}")
        return

    print(f"Converting dataset at: {source_dir}")
    hdf5_file = h5py.File(hdf5_path, mode="w")
    sub_dirs = list(os.walk(source_dir))
    file_count = sum(len(files) for (_, _, files) in sub_dirs)

    with tqdm(total=file_count) as pbar:
        for root, _, files in sub_dirs:
            g_name = root.replace(source_dir, "")
            g = hdf5_file.create_group(g_name) if g_name else hdf5_file
            for f in files:
                filepath = os.path.join(root, f)
                if os.path.isfile(filepath):
                    with open(filepath, "rb") as fp:
                        file_content = fp.read()
                    g.create_dataset(
                        f, data=np.frombuffer(file_content, dtype="uint8")
                    )
                pbar.update()

    hdf5_file.close()
    print("done.")


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Converts a dataset at the specified path to hdf5. The "
        "local directory structure is preserved in the hdf5 file."
    )
    parser.add_argument(
        "-p",
        "--path",
        required=True,
        help="path to the root folder of a specific dataset to convert",
    )
    args = parser.parse_args()
    convert_dataset(args.path)
