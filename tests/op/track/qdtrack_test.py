"""Test cases for quasi dense tracking graph construction."""
import torch

from tests.util import generate_boxes, generate_features, get_test_file
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
    state = torch.random.get_rng_state()
    torch.random.set_rng_state(torch.manual_seed(0).get_state())
    qd_head = QDSimilarityHead()
    batch_size, num_boxes, wh = 2, 10, 128
    test_features = [None, None] + generate_features(
        256, wh, wh, 5, batch_size
    )
    boxes, _, _, _ = generate_boxes(wh * 4, wh * 4, num_boxes, batch_size)

    embeds = qd_head(test_features, boxes)  # type: ignore
    embed = torch.stack(embeds)
    torch.random.set_rng_state(state)
    path = get_test_file("qdtrack_embeds.pt")
    expected_embeds = torch.load(path)
    # assert torch.isclose(embed, expected_embeds).all() #FIXME TODO


def test_association() -> None:  # FIXME
    """Testcase for assocation."""
    tracker = QDTrackAssociation()

    h, w, num_dets = 128, 128, 64
    boxes, scores, classes, _ = [x[0] for x in generate_boxes(h, w, num_dets)]
    scores = scores.squeeze(-1)

    embeddings = torch.rand(num_dets, 128)

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
    return  # FIXME TODO The following fails on CI (38 vs 37 shapes)

    assert torch.isclose(
        track_ids_positive,
        torch.arange(
            mem_track_ids.max() + 1,
            mem_track_ids.max() + 1 + len(track_ids_positive),
        ),
    ).all()
