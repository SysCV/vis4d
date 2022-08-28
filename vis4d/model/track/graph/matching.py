import torch


def calc_bisoftmax_affinity(
    detection_class_ids: torch.Tensor,
    detection_embeddings: torch.Tensor,
    track_class_ids: torch.Tensor,
    track_embeddings: torch.Tensor,
    with_categories: bool = True,
) -> torch.Tensor:
    """Calculate affinity matrix using bisoftmax metric."""
    feats = torch.mm(detection_embeddings, track_embeddings.t())
    d2t_scores = feats.softmax(dim=1)
    t2d_scores = feats.softmax(dim=0)
    similarity_scores = (d2t_scores + t2d_scores) / 2

    if with_categories:
        cat_same = detection_class_ids.view(-1, 1) == track_class_ids.view(
            1, -1
        )
        similarity_scores *= cat_same.float()
    return similarity_scores
