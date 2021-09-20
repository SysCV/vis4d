"""Nearest Neighbor metric."""
from typing import Dict, List, Optional

import torch


def _cosine_distance(
    a: List[torch.tensor],
    b: List[torch.tensor],
    data_is_normalized: bool = False,
) -> torch.tensor:  # pylint: disable = invalid-name
    """Compute pair-wise cosine distance between points in `a` and `b`.

    Args:
        a : An NxL matrix of N samples of dimensionality L.
        b :  An MxL matrix of M samples of dimensionality L.
        data_is_normalized : If True, assumes rows in a and b are unit length
            vectors. Otherwise, a and b are explicitly normalized to lenght 1.

    Returns:
        Returns a matrix of size NxM such that element (i, j)
        contains the squared distance between `a[i]` and `b[j]`.

    """
    if not data_is_normalized:
        samples_a = torch.stack(a, dim=0)
        normed_a = samples_a / torch.linalg.norm(
            samples_a, dim=1, keepdims=True
        )
        samples_b = torch.stack(b, dim=0)
        normed_b = samples_b / torch.linalg.norm(
            samples_b, dim=1, keepdims=True
        )
    return 1.0 - torch.matmul(normed_a, normed_b.T)


def _nn_cosine_distance(
    x: List[torch.tensor], y: List[torch.tensor]
) -> torch.tensor:  # pylint: disable = invalid-name
    """Helper function for nearest neighbor distance metric (cosine).

    Args:
        x : A matrix of N row-vectors (sample points).
        y : A matrix of M row-vectors (query points).

    Returns:
        min_distances: A vector of length M that contains for each entry in `y`
        the smallest cosine distance to a sample in `x`.
    """
    distances = _cosine_distance(x, y)
    min_distance = torch.min(distances, dim=0)[0]
    return min_distance


class NearestNeighborDistanceMetric:
    """A nearest neighbor distance metric.

    For each target, returns the closest distance to any sample that has been
    observed so far.

    Args:
        matching_threshold: float
            The matching threshold. Samples with larger distance are considered
            an invalid match.
        budget : Optional[int]
            If not None, fix samples per class to at most this number. Removes
            the oldest samples when the budget is reached.
        samples : Dict[int -> List[ndarray]]
            A dictionary that maps from target identities to the list of
            samples that have been observed so far.
    """

    def __init__(
        self,
        matching_threshold: float,
        budget: Optional[int] = None,
    ):
        """Init."""
        self._metric = _nn_cosine_distance
        self.matching_threshold = matching_threshold
        self.budget = budget
        self.samples: Dict[int, List[torch.tensor]] = {}

    def partial_fit(
        self,
        features: List[torch.tensor],
        targets: List[int],
        active_targets: List[int],
    ) -> None:
        """Update the distance metric with new data.

        Args:
            features: An NxM matrix of N features of dimensionality M.
            targets: An integer tensor of associated target identities.
            active_targets: A list of targets that are currently present in the
                scene.
        """
        for feature, target in zip(features, targets):
            self.samples.setdefault(target, []).append(feature)
            if self.budget is not None:
                self.samples[target] = self.samples[target][-1 * self.budget :]
        self.samples = {k: self.samples[k] for k in active_targets}

    def distance(
        self, features: List[torch.tensor], targets: List[int]
    ) -> torch.tensor:
        """Compute distance between features and targets.

        Args:
            features :An NxL matrix of N features of dimensionality L.
            targets : A list of targets to match the given `features` against.

        Returns:
            cost_matrix: a cost matrix of shape [len(targets), len(features)],
            where element (i, j) contains the closest squared distance between
            `targets[i]` and `features[j]`.
        """
        cost_matrix = torch.empty((0, len(features))).to(features[0].device)
        # cost_matrix = torch.zeros((len(targets), len(features)))
        for _, target in enumerate(targets):
            min_dist = self._metric(self.samples[target], features)
            cost_matrix = torch.cat(
                (cost_matrix, min_dist.unsqueeze(0)), dim=0
            )
        return cost_matrix
