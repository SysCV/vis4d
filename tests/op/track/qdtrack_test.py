"""Test cases for quasi dense tracking graph construction."""
import torch

from tests.util import get_test_file
from vis4d.op.track.assignment import TrackIDCounter
from vis4d.op.track.qdtrack import (
    QDSimilarityHead,
    QDTrackAssociation,
    QDTrackInstanceSimilarityLoss,
)


def test_qdtrack_loss() -> None:
    """Test loss implementation."""
    loss = QDTrackInstanceSimilarityLoss()
    key_embeds = [torch.ones((10, 128))]
    ref_embeds = [[torch.ones((10, 128))]]
    key_track_ids = [torch.zeros((10,))]
    ref_track_ids = [[torch.zeros((10,))]]
    loss_values = loss(key_embeds, ref_embeds, key_track_ids, ref_track_ids)
    assert all(torch.isclose(v, torch.tensor(0.0)) for v in loss_values)


def test_similarity_head() -> None:
    """Testcase for similarity head."""
    qd_head = QDSimilarityHead()
    path = get_test_file("qd_head_weights.pt")
    qd_head.load_state_dict(torch.load(path))
    path = get_test_file("qdtrack_embeds.pt")
    test_features, boxes, expected_embeds = torch.load(path)
    embeds = qd_head(test_features, boxes)
    embed = torch.stack(embeds)
    assert torch.isclose(embed, expected_embeds).all()


def test_association() -> None:
    """Testcase for assocation."""
    tracker = QDTrackAssociation()

    path = get_test_file("qdtrack_association_inputs.pt")
    boxes, scores, classes, embeddings = torch.load(path)

    # feed same detections & embeddings --> should be matched to self
    mem_track_ids = TrackIDCounter.get_ids(len(boxes))
    mem_embeddings = embeddings.clone()
    mem_classes = classes.clone()

    track_ids, indices = tracker(
        boxes,
        scores,
        classes,
        embeddings,
        mem_track_ids,
        mem_classes,
        mem_embeddings,
    )
    scores_permute = scores[indices]
    track_ids_negative = track_ids[scores_permute < tracker.obj_score_thr]
    track_ids_positive = track_ids[scores_permute >= tracker.obj_score_thr]
    assert torch.isclose(
        track_ids_negative, torch.tensor(-1, dtype=torch.long)
    ).all()
    assert torch.isclose(
        track_ids_positive, mem_track_ids[indices[track_ids != -1]]
    ).all()

    # test non-equal classes
    track_ids, indices = tracker(
        boxes,
        scores,
        classes + 1,
        embeddings,
        mem_track_ids,
        mem_classes,
        mem_embeddings,
    )
    track_ids_negative = track_ids[scores_permute < tracker.init_score_thr]
    track_ids_positive = track_ids[scores_permute >= tracker.init_score_thr]

    assert torch.isclose(
        track_ids_negative, torch.tensor(-1, dtype=torch.long)
    ).all()
    assert torch.isclose(
        track_ids_positive,
        torch.arange(
            mem_track_ids.max() + 1,
            mem_track_ids.max() + 1 + len(track_ids_positive),
        ),
    ).all()
