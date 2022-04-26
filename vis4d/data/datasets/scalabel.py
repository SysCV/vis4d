"""Dataset loader for scalabel format."""

from scalabel.label.io import load, load_label_config
from scalabel.label.typing import Dataset

from .base import BaseDatasetLoader


class Scalabel(BaseDatasetLoader):
    """Scalabel dataloading class."""

    def load_dataset(self) -> Dataset:
        """Load Scalabel frames from json."""
        assert self.annotations is not None
        dataset = load(self.annotations, nprocs=self.num_processes)
        metadata_cfg = dataset.config
        if self.config_path is not None:
            metadata_cfg = load_label_config(self.config_path)
        assert metadata_cfg is not None
        dataset.config = metadata_cfg
        return dataset





def predictions_to_scalabel(
    predictions: Dict[str, List[TLabelInstance]],
    idx_to_class: Optional[Dict[int, str]] = None,
) -> ModelOutput:  # TODO integrate
    """Convert predictions into ModelOutput (Scalabel)."""
    outputs = {}
    for key, values in predictions.items():
        outputs[key] = [
            v.to(torch.device("cpu")).to_scalabel(idx_to_class) for v in values
        ]
    return outputs