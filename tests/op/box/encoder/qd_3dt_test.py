"""QD3DT Box3D encoder decoder test file."""
import torch

from vis4d.op.box.encoder.qd_3dt import QD3DTBox3DEncoder


def test_qd_3dt_encode() -> None:
    """Test QD3DT box3d encoder."""
    encoder = QD3DTBox3DEncoder()

    boxes = torch.tensor(
        [[0, 0, 100, 100], [0, 0, 100, 100]], dtype=torch.float32
    )
    boxes3d = torch.tensor(
        [
            [10, 10, 1, 100, 100, 100, 0, 0, 0, 0],
            [30, 30, 1, 100, 100, 100, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    intrinsics = torch.tensor(
        [[100, 0, 50], [0, 100, 50], [0, 0, 1]], dtype=torch.float32
    )

    enc = encoder(boxes, boxes3d, intrinsics)

    assert enc.shape == (2, 10)
    assert torch.isclose(
        enc[0],
        torch.tensor(
            [
                100.0000,
                100.0000,
                0.0000,
                9.2103,
                9.2103,
                9.2103,
                0.0000,
                0.0000,
                0.0000,
                0.0000,
            ]
        ),
    ).all()
