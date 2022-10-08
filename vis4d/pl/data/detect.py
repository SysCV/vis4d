from typing import List

from torch.utils.data import DataLoader

from vis4d.data.datasets import COCO
from vis4d.run.data.detect import default_train_pipeline, default_test_pipeline

from .base import DataModule


class DetectDataModule(DataModule):
    """Detect data module."""

    def train_dataloader(self) -> DataLoader:
        """Setup training data pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO("data/COCO/", data_backend=data_backend)
            dataloader = default_train_pipeline(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloader

    def test_dataloader(self) -> List[DataLoader]:
        """Setup inference pipeline."""
        data_backend = self._setup_backend()
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            dataset = COCO(
                "data/COCO/", data_backend=data_backend, split="val2017"
            )
            dataloaders = default_test_pipeline(
                dataset, self.samples_per_gpu, (800, 1333)
            )
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return dataloaders

    def evaluators(self) -> List[Evaluator]:
        """Define evaluators associated with test datasets."""
        if self.experiment == "bdd100k":
            raise NotImplementedError
        elif self.experiment == "coco":
            evaluators = [COCOEvaluator("data/COCO/", split="val2017")]
        else:
            raise NotImplementedError(
                f"Experiment {self.experiment} not known!"
            )
        return evaluators


# class InsSegDataModule(DataModule): TODO revise
#     """InsSeg data module."""

#     def train_dataloader(self) -> DataLoader:
#         """Setup training data pipeline."""
#         data_backend = self._setup_backend()
#         if self.experiment == "bdd100k":
#             raise NotImplementedError
#         elif self.experiment == "coco":
#             dataset = COCO(
#                 "data/COCO/", with_mask=True, data_backend=data_backend
#             )
#             dataloader = default_train(
#                 dataset, self.samples_per_gpu, (800, 1333)
#             )
#         else:
#             raise NotImplementedError(
#                 f"Experiment {self.experiment} not known!"
#             )
#         return dataloader

#     def test_dataloader(self) -> List[DataLoader]:
#         """Setup inference pipeline."""
#         data_backend = self._setup_backend()
#         if self.experiment == "bdd100k":
#             raise NotImplementedError
#         elif self.experiment == "coco":
#             dataset = COCO(
#                 "data/COCO/",
#                 with_mask=True,
#                 split="val2017",
#                 data_backend=data_backend,
#             )
#             dataloaders = default_test(
#                 dataset, self.samples_per_gpu, (800, 1333)
#             )
#         else:
#             raise NotImplementedError(
#                 f"Experiment {self.experiment} not known!"
#             )
#         return dataloaders

#     def evaluators(self) -> List[Evaluator]:
#         """Define evaluators associated with test datasets."""
#         if self.experiment == "bdd100k":
#             raise NotImplementedError
#         elif self.experiment == "coco":
#             evaluators = [
#                 COCOEvaluator("data/COCO/", split="val2017"),
#                 COCOEvaluator("data/COCO/", iou_type="segm", split="val2017"),
#             ]
#         else:
#             raise NotImplementedError(
#                 f"Experiment {self.experiment} not known!"
#             )
#         return evaluators
