"""Track assignment functions."""
import torch


def greedy_assign(
    detection_scores: torch.Tensor,
    tracklet_ids: torch.Tensor,
    affinity_scores: torch.Tensor,
    match_score_thr: float = 0.5,
) -> torch.Tensor:
    """Greedy assignment of detections to tracks given affinities."""
    ids = torch.full(
        (len(detection_scores),),
        -1,
        dtype=torch.long,
        device=detection_scores.device,
    )

    for i in range(len(detection_scores)):
        conf, memo_ind = torch.max(affinity_scores[i, :], dim=0)
        cur_id = tracklet_ids[memo_ind]
        if conf > match_score_thr:
            if cur_id != -1:
                ids[i] = cur_id
                affinity_scores[:i, memo_ind] = 0
                affinity_scores[(i + 1) :, memo_ind] = 0

    return ids


def random_ids(
    num_ids: int, device: torch.device = torch.device("cpu")
) -> torch.Tensor:
    """Generate a num_ids number of unique ids.

    Args:
        num_ids (int): number of ids
        device (torch.device, optional): Device to create ids on. Defaults to torch.device("cpu").

    Returns:
        torch.Tensor: Tensor of random ids with 64 bit. Collision probability
        when generating 19208 ids: 1 in 100 billion.
    """
    return torch.randint(
        torch.iinfo(torch.int64).min,
        torch.iinfo(torch.int64).max,
        (num_ids,),
        device=device,
    )
