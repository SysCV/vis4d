"""Test data for mask rcnn operators."""

import torch

INSSEG0_MASKS = torch.tensor(
    [
        [0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    ],
    dtype=torch.uint8,
)
INSSEG0_INDICES = [
    torch.tensor([[0], [1], [2], [3]]),
    torch.stack(
        [
            torch.arange(20),
            torch.arange(250, 270),
            torch.arange(210, 230),
            torch.arange(190, 210),
        ]
    ),
    torch.stack(
        [
            torch.arange(110, 130),
            torch.arange(300, 320),
            torch.arange(265, 285),
            torch.arange(440, 460),
        ]
    ),
]
INSSEG0_SCORES = torch.tensor([0.9937, 0.9919, 0.6182, 0.5119])
INSSEG0_CLASS_IDS = torch.tensor([75, 15, 73, 73])

INSSEG1_MASKS = torch.tensor(
    [
        [0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    ],
    dtype=torch.uint8,
)
INSSEG1_INDICES = [
    torch.tensor([[0], [1], [2], [3]]),
    torch.stack(
        [
            torch.arange(280, 300),
            torch.arange(260, 280),
            torch.arange(320, 340),
            torch.arange(290, 310),
        ]
    ),
    torch.stack(
        [
            torch.arange(275, 295),
            torch.arange(240, 260),
            torch.arange(350, 370),
            torch.arange(440, 460),
        ]
    ),
]
INSSEG1_SCORES = torch.tensor(
    [0.8863, 0.8347, 0.7669, 0.7414, 0.6059, 0.5854, 0.5791]
)
INSSEG1_CLASS_IDS = torch.tensor([57, 56, 57, 58, 73, 56, 73])
