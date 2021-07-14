"""Pack a dataset folder into an hdf5 file."""

import argparse
import os
from argparse import Namespace

import h5py
import numpy as np
from tqdm import tqdm

from openmt.config.config import read_config

t_vlen_uint8 = h5py.special_dtype(vlen=np.uint8)


def datasets_to_hdf5(args: Namespace) -> None:
    """Convert datasets to hdf5."""
    if args.config is not None:
        cfg = read_config(args.config)
        if cfg.train is not None:
            for dataset in cfg.train:
                convert_single_dataset(dataset.data_root)

        if cfg.test is not None:
            for dataset in cfg.test:
                convert_single_dataset(dataset.data_root)

    if args.path is not None:
        convert_single_dataset(args.path)


def convert_single_dataset(source_dir: str) -> None:
    """Convert particular dataset instance to hdf5."""
    print(f"Converting dataset at: {source_dir}")
    hdf5_path = source_dir.rstrip("/") + ".hdf5"
    if os.path.exists(hdf5_path):
        print(f"File {hdf5_path} already exists! Skipping {source_dir}")
        return
    hdf5_file = h5py.File(hdf5_path, mode="w")

    file_count = sum(len(files) for _, _, files in os.walk(source_dir))
    with tqdm(total=file_count) as pbar:
        for root, dirs, files in os.walk(source_dir):
            if not dirs:
                if root is not source_dir:
                    group = hdf5_file.create_group(root.split("/")[-1])
                else:
                    group = hdf5_file
                for f in files:
                    filepath = os.path.join(root, f)
                    if os.path.isfile(filepath):
                        with open(filepath, "rb") as fp:
                            file_content = fp.read()
                        group.create_dataset(
                            f,
                            data=np.frombuffer(  # type: ignore
                                file_content, dtype="uint8"
                            ),
                        )
                    pbar.update()

    hdf5_file.close()
    print("done.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Converts all datasets in a specific config file to hdf5 "
        "format. Alternatively, converts a specific dataset at "
        "the specified path to hdf5."
    )
    parser.add_argument(
        "-c",
        "--config",
        default=None,
        help="path to config file which contains info of datasets",
    )
    parser.add_argument(
        "-p",
        "--path",
        default=None,
        help="path to the root folder of a specific dataset to convert",
    )
    datasets_to_hdf5(parser.parse_args())
