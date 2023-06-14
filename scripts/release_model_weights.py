from __future__ import annotations

import argparse
import os
import hashlib
import torch
from pytorch_lightning import LightningModule


def load_lightning_model(filepath: str) -> dict[str, torch.Tensor]:
    """
    Load a PyTorch Lightning model from a checkpoint.

    Args:
        filepath (str): The path to the PyTorch Lightning checkpoint.

    Returns:
        model (LightningModule): The loaded PyTorch Lightning model.
    """
    checkpoint = torch.load(filepath, map_location=torch.device("cpu"))
    return checkpoint


def release_model_weights(
    state_dict: dict[str, torch.Tensor], path: str, filename: str
) -> None:
    """
    Saves the model weights and append a 6-digit hash to the filename.

    Args:
        state_dict (dict[str, torch.Tensor]): The model weights to save.
        path (str): The directory path to save the model.
        filename (str): The filename to save the model.
    """
    os.makedirs(path, exist_ok=True)
    with open(os.path.join(path, filename), "wb") as f:
        torch.save(state_dict, f)

    # Create a hash of the file
    sha256_hash = hashlib.sha256()
    with open(os.path.join(path, filename), "rb") as f:
        # Read and update hash in chunks of 4K
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)

    # Get the hexadecimal representation of the hash
    short_hash = sha256_hash.hexdigest()[:6]
    os.rename(
        os.path.join(path, filename),
        os.path.join(path, f"{filename}_{short_hash}.pth"),
    )


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Save trained model checkpoint with a filename hash."
    )
    parser.add_argument("path", type=str, help="The path to the checkpoint.")
    parser.add_argument(
        "--outdir",
        type=str,
        help="The path to output the model.",
        default=".vis4d-workspace/release",
    )
    parser.add_argument(
        "--name", type=str, help="The path to output the model."
    )
    args = parser.parse_args()

    state_dict = load_lightning_model(args.path)
    release_model_weights(state_dict, args.outdir, args.name)


if __name__ == "__main__":
    main()
