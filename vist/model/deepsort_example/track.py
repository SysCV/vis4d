"""DeepSORT track."""
from typing import Optional

import torch

from .detection import Detection
from .kalman_filter import KalmanFilter


class TrackState:
    """Track state.

    Newly created tracks are classified as `tentative` until enough evidence
    has been collected. Then, the track state is changed to `confirmed`.
    Tracks that are no longer alive are classified as `deleted` to mark them
    for removal from the set of active tracks.
    """

    Tentative = 1
    Confirmed = 2
    Deleted = 3


class Track:
    """class to store a single track for one tracking instance.

    Attributes
    ----------
    mean : torch.tensor
        The 8-dimensional state space, (x, y, a, h, vx, vy, va, vh), contains
        the bounding box center position (x, y), aspect ratio a, height h,
        and their respective velocities.
    confidence: float
        bounding box detection confidence score
    covariance : torch.tensor
        Covariance matrix of the state distribution.
    track_id : int
        A unique track identifier.
    n_init : int
        Number of consecutive detections before the track is confirmed. The
        track state is set to `Deleted` if a miss occurs within the first
        `n_init` frames.
    max_age : int
        The maximum number of consecutive misses before the track state is
        set to `Deleted`.
    feature : Optional[torch.tensor]
        Feature vector of the detection.
    hits : int
        Total number of measurement updates.
    age : int
        Total number of frames since first occurance.
    time_since_update : int
        Total number of frames since last measurement update.
    state: TrackState
        The current track state.
    features : List[torch.tensor]
        A cache of features. On each measurement update, the associated feature
        vector is added to this list.
    """

    def __init__(
        self,
        mean: torch.tensor,
        covariance: torch.tensor,
        confidence: float,
        class_id: int,
        track_id: int,
        n_init: int,
        max_age: int,
        feature: Optional[torch.tensor] = None,
    ):
        """Init."""
        self.mean = mean
        self.covariance = covariance
        self.confidence = confidence
        self.class_id = class_id
        self.track_id = track_id
        self.hits = 1
        self.age = 1
        self.time_since_update = 0

        # self.state = TrackState.Tentative
        self.state = TrackState.Confirmed
        self.features = []
        if feature is not None:
            self.features.append(feature)

        self._n_init = n_init
        self._max_age = max_age

    def to_tlwh(self):
        """Get bounding box in `(top left x, top left y, width, height)."""
        ret = self.mean[:4].clone().detach()
        ret[2] *= ret[3]
        ret[:2] -= ret[2:] / 2.0
        return ret

    def to_tlbr(self):
        """Get bounding box in `(min x, miny, max x, max y)`."""
        ret = self.to_tlwh()
        ret[2:] = ret[:2] + ret[2:]
        return ret

    def predict(self, kf: KalmanFilter):
        """State prediction to the current time.

        Propagate the state distribution to the current time step using a
        Kalman filter prediction step.

        Args:
            kf: KalmanFilter
        """
        self.mean, self.covariance = kf.predict(
            self.mean, self.covariance, self.class_id
        )
        self.age += 1
        self.time_since_update += 1

    def update(self, kf: KalmanFilter, detection: Detection):
        """Kalman filter measurement update step and update the feature cache."""  # pylint: disable=line-too-long
        self.mean, self.covariance = kf.update(
            self.mean, self.covariance, detection.to_xyah(), self.class_id
        )
        self.features.append(detection.feature)
        self.confidence = detection.confidence
        self.hits += 1
        self.time_since_update = 0
        if self.state == TrackState.Tentative and self.hits >= self._n_init:
            self.state = TrackState.Confirmed

        self.mean[:4] = detection.to_xyah()

    def mark_missed(self):
        """Mark this track as missed.

        no association at the current time step.
        """
        if self.state == TrackState.Tentative:
            self.state = TrackState.Deleted
        elif self.time_since_update > self._max_age:
            self.state = TrackState.Deleted

    def is_tentative(self):
        """Returns True if this track is tentative (unconfirmed)."""
        return self.state == TrackState.Tentative

    def is_confirmed(self):
        """Returns True if this track is confirmed."""
        return self.state == TrackState.Confirmed

    def is_deleted(self):
        """Returns True if this track is dead and should be deleted."""
        return self.state == TrackState.Deleted
