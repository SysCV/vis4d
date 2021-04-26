"""Track graph of SORT."""
from openmt.struct import Boxes2D
from openmt.model.track.graph import BaseTrackGraph, TrackGraphConfig


class SORTTrackGraphConfig(TrackGraphConfig):
    example_additional_attr: str = "hello"


class SORTTrackGraph(BaseTrackGraph):
    """SORT tracking logic."""

    def __init__(self, cfg: TrackGraphConfig) -> None:
        """Init."""
        super().__init__(cfg)
        self.cfg = SORTTrackGraphConfig(**cfg.dict())

    def forward(  # type: ignore
        self, detections: Boxes2D, frame_id: int, *args, **kwargs
    ) -> Boxes2D:
        """Process inputs, match detections with existing tracks."""
        raise NotImplementedError  # TODO implement sort tracking logic

    def update(self, *args, **kwargs) -> None:  # type: ignore
        """Update track memory using matched detections."""
        raise NotImplementedError
