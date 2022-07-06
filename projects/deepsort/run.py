"""
  Example implementation of the Deep Sort tracking algorithm:
  https://github.com/nwojke/deep_sort
"""

# Dataset
from projects.common.datasets import bdd100k_det_map

# Model
from projects.common.models import build_retinanet
from projects.deepsort.data import DetectDataModule

# Core
from vis4d.engine.trainer import BaseCLI, DefaultTrainer
from vis4d.model.deepsort import DeepSort
from vis4d.model.track.graph.deepsort import DeepSORTTrackGraph
from vis4d.model.track.similarity import DeepSortSimilarityHead


def setup_model(
    experiment: str,
    lr: float = 0.003,
    optimizer: str = "torch.optim.Adam",
    detector: str = "Retinanet",
    freeze_detector: bool = False,
) -> DeepSort:
    """Setup model with experiment specific hyperparameters."""
    if experiment == "bdd100k":
        category_mapping = bdd100k_det_map
    else:
        raise NotImplementedError(f"Experiment {experiment} not known!")

    # This defines the same kalman filter parameter for all classes.
    # If you would like to use different parameters for each class,
    # the parameter
    # will need to be augmented as follows:
    # kalman_filter_params = {0:{  "cov_motion_Q": ...,..},
    # 1: {"cov_motion_Q": ...,..}, ...}
    # where the dict needs to have an entry for each semantic class

    kalman_filter_params = {
        # Motion covariance matrix (6x6 for bbox position + velocity)
        "cov_motion_Q": [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1],
        # Measurement noise covariance matrix (4x4 since we only measure
        # bbox location not velocity)
        "cov_project_R": [0.1, 0.1, 0.1, 0.1],
        # Initial covariance matrix
        "cov_P0": [1, 1, 1, 1, 1, 1, 1, 1],
    }

    if detector == "Retinanet":
        detection_model = build_retinanet(category_mapping)
    else:
        raise NotImplementedError(
            f"Invalid detector {detector}. Currently only Retinanet can be "
            f"used as detector!"
        )

    # Freeze detector and only train deep sort similarity head
    if freeze_detector:
        detection_model.freeze()

    return DeepSort(
        detection=detection_model,
        # What similarity head to use to extract feature representation for
        # detections
        similarity=DeepSortSimilarityHead(
            num_classes=len(category_mapping.keys())
        ),
        # What track graph to use to associate and manage tracklets
        track_graph=DeepSORTTrackGraph(
            num_classes=len(category_mapping.keys()),
            kalman_filter_params=kalman_filter_params,
        ),
        category_mapping=category_mapping,
        optimizer_init={
            "class_path": optimizer,
            "init_args": {"lr": lr},
        },
    )


class SegmentCLI(BaseCLI):
    """Segment CLI."""

    def add_arguments_to_parser(self, parser):
        """Link data and model experiment argument."""
        parser.link_arguments("data.experiment", "model.experiment")
        parser.link_arguments("model.max_steps", "trainer.max_steps")
        parser.link_arguments("trainer.gpus", "model.gpus")


if __name__ == "__main__":
    SegmentCLI(
        model_class=setup_model,
        datamodule_class=DetectDataModule,
        trainer_class=DefaultTrainer,
    )
