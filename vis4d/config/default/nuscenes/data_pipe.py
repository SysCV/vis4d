"""Data pipe for nuScenes dataset."""
from vis4d.common.typing import ArgsType
from vis4d.data.datasets import VideoMixin
from vis4d.data.loader import DataPipe


class NuscVideoDataPipe(DataPipe, VideoMixin):  # TODO: refactor data pipe
    def __init__(self, *args: ArgsType, **kwargs: ArgsType) -> None:
        super().__init__(*args, **kwargs)
        assert len(self.datasets) == 1, "Only support one dataset for now."

    @property
    def video_to_indices(self):
        return self.datasets[0].video_to_indices
