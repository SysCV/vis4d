"""BEVFormer model zoo."""

from . import bevformer_base, bevformer_tiny, bevformer_vis

AVAILABLE_MODELS = {
    "bevformer_base": bevformer_base,
    "bevformer_tiny": bevformer_tiny,
    "bevformer_vis": bevformer_vis,
}
