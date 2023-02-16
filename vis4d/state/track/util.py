"""Utilities for tracking state."""
from __future__ import annotations

import torch
from torch import Tensor
from .base import TTrackState
import pdb


def update_frames(
    frames: list[TTrackState], data: TTrackState, memory_limit: int
) -> list[TTrackState]:
    """Store tracks in memory.

    Args:
        frames (list[TTrackState]): List of frames.
        data (TTrackState): Data to be stored.
        memory_limit (int): Frame limit of memory.
    """
    frames.append(data)
    if memory_limit >= 0 and len(frames) > memory_limit:
        frames.pop(0)
    return frames


def concat_states(states: list[TTrackState]) -> tuple[Tensor, ...]:
    """Concatenate states.

    Args:
        states (list[TTrackState]): List of states.

    Returns:
        memory_states (tuple[Tensor, ...]): Concatenated states.
    """
    memory_states = tuple(torch.cat(state) for state in zip(*states))
    return memory_states


def merge_tracks(tracks_1: TTrackState, tracks_2: TTrackState) -> TTrackState:
    """Merge two tracks.

    Args:
        tracks_1 (TTrackState): First track.
        tracks_2 (TTrackState): Second track.

    Returns:
        merged_tracks (TTrackState): Merged tracks.
    """
    merged_tracks = tuple(
        torch.cat((track_1, track_2))
        for track_1, track_2 in zip(tracks_1, tracks_2)
    )
    return merged_tracks


def get_last_tracks(memory_states: TTrackState) -> tuple[Tensor, ...]:
    """Get last states of all tracks.

    Args:
        memory_states (TTrackState): Memory states.

    Returns:
        last_tracks (tuple[Tensor, ...]): Last tracks.
    """
    track_ids = memory_states.track_ids.unique()
    value_dict = {k: [] for k in memory_states._fields}
    for track_id in track_ids:
        idx = (memory_states.track_ids == track_id).nonzero(as_tuple=False)[-1]
        for k in value_dict:
            value_dict[k].append(getattr(memory_states, k)[idx])

    last_tracks = (torch.cat(value_dict[k]) for k in value_dict)
    return last_tracks
