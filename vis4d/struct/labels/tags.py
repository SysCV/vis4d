"""Vis4D Image Tags data structure."""
from typing import Dict, List, Optional

import numpy as np
import torch
from scalabel.label.typing import Frame

from vis4d.struct.structures import LabelInstance


class ImageTags(LabelInstance):
    """Container class for image tags.

    tags: torch.LongTensor (N,) where each entry is a image tag and N is the
    number of labels.
    attribute: list of attribute names.
    """

    def __init__(self, tags: torch.Tensor, attribute: List[str]) -> None:
        """Init."""
        assert isinstance(tags, torch.Tensor) and len(tags.shape) == 1
        self.tags = tags
        self.attribute = attribute

    @classmethod
    def empty(cls, device: Optional[torch.device] = None) -> "ImageTags":
        """Return empty tags on device."""
        return ImageTags(torch.empty(0), [""]).to(device)

    @classmethod
    def merge(cls, instances: List["ImageTags"]) -> "ImageTags":
        """Merges a list of ImageTags into a single ImageTags."""
        assert isinstance(instances, (list, tuple))
        assert len(instances) > 0
        assert all((isinstance(inst, ImageTags) for inst in instances))
        assert all(instances[0].device == inst.device for inst in instances)

        tags, attrs = [], []
        for t in instances:
            tags.append(t.tags)
            attrs.extend(t.attribute)
        return cls(torch.cat(tags), attrs)

    @classmethod
    def from_scalabel(  # type: ignore
        cls,
        frame: Frame,
        class_to_idx: Dict[str, int],
        attribute: Optional[str],
    ) -> "ImageTags":
        """Convert from scalabel format to internal."""
        if (
            frame.attributes is None or len(frame.attributes) == 0
        ):  # pragma: no cover
            return ImageTags.empty()
        if attribute is None:  # pragma: no cover
            # get default attribute
            assert len(frame.attributes) == 1, (
                "Number of attributes in frame should only be 1 if no"
                " attribute is specified"
            )
            tag_attr = list(frame.attributes.keys())[0]
        else:
            tag_attr = attribute
        attr_cls = frame.attributes[tag_attr]
        assert isinstance(attr_cls, str)
        attr_idx = class_to_idx[attr_cls]
        tag_tensor = torch.tensor(np.array([attr_idx]), dtype=torch.int64)
        return ImageTags(tag_tensor, [tag_attr])

    def to_scalabel(
        self, idx_to_class: Optional[Dict[int, str]] = None
    ) -> Frame:
        """Convert from internal to scalabel format."""
        assert idx_to_class is not None
        return Frame(
            name="",
            attributes={self.attribute[0]: idx_to_class[self.tags[0].item()]},
        )

    def __getitem__(self, item) -> "ImageTags":  # type: ignore
        """Shadows tensor based indexing while returning new ImageTags."""
        if isinstance(item, tuple):  # pragma: no cover
            item = item[0]
        tags, attr = self.tags[item], [self.attribute[item]]
        if len(tags.shape) < 1:
            return ImageTags(tags.view(1), attr)
        return ImageTags(tags, attr)  # pragma: no cover

    def __len__(self) -> int:
        """Get length of the object."""
        return len(self.tags)

    def clone(self) -> "ImageTags":
        """Create a copy of the object."""
        return ImageTags(self.tags.clone(), self.attribute)

    def to(self, device: torch.device) -> "ImageTags":
        """Move data to given device."""
        return ImageTags(self.tags.to(device=device), self.attribute)

    @property
    def device(self) -> torch.device:
        """Get current device of data."""
        return self.tags.device
