"""Track momory base classes."""
from __future__ import annotations

from typing import Generic, NamedTuple, TypeVar

from torch import device

# TODO refactor this from inheritance to composition
# pylint: disable=invalid-name
TTrackState = TypeVar("TTrackState", bound=NamedTuple)


class BaseTrackMemory(Generic[TTrackState]):
    """Basic Track Memory class.

    Holds track representation across timesteps as list[NamedTuple].
    Each list element is tracks at time t and tracks at time t are
    represented as NamedTuple, where the first element is a LongTensor of track
    ids, and the other N elements are, e.g., boxes, scores, class_ids, etc.
    We assume the memory works on frames with contiguous indices [0..N].
    """

    def __init__(self, memory_limit: int = -1):
        """Creates an instance of the class.

        Args:
            memory_limit (int, optional): Frame limit of memory.
                Defaults to -1.
        """
        assert memory_limit >= -1
        self.memory_limit = memory_limit
        self.frames: list[TTrackState] = []

    def reset(self) -> None:
        """Empty the memory."""
        self.frames = []

    @property
    def last_frame(self) -> TTrackState:
        """Return last frame stored in memory.

        Returns:
            TTrackState: Last frame representation.
        """
        return self.frames[-1]

    def get_frame(self, index: int) -> TTrackState:
        """Get TrackState at frame with given index.

        Returns:
            TTrackState: frame representation at index.
        """
        return self.frames[index]

    def get_frames(
        self, start_index: int, end_index: int
    ) -> list[TTrackState]:
        """Get list of frames at certain time interval.

        Args:
            start_index (int): start index of interval (incl).
            end_index (int): end index of interval (excl).

        Returns:
            list[TTrackState]: frame representations inside interval.
        """
        return self.frames[start_index:end_index]

    def get_current_tracks(
        self, device: device  # pylint: disable=unused-argument
    ) -> TTrackState:
        """Return active tracks."""
        return self.frames[-1]

    def update(self, data: TTrackState) -> None:
        """Store tracks in memory."""
        self.frames.append(data)
        if self.memory_limit >= 0 and len(self.frames) > self.memory_limit:
            self.frames.pop(0)
