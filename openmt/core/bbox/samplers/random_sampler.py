from detectron2.modeling.sampling import subsample_labels

from .base_sampler import BaseSampler


class RandomSampler(BaseSampler):
    def __init__(
        self,
        batch_size_per_image,
        positive_fraction,
        num_classes,
        proposal_matcher,
        proposal_append_gt,
    ):

        # TODO num classes really needed?

        self.batch_size_per_image = batch_size_per_image
        self.positive_fraction = positive_fraction
        self.num_classes = num_classes
        self.proposal_matcher = proposal_matcher
        self.proposal_append_gt = proposal_append_gt

    def sample(self, boxes, labels):
        """Sample boxes randomly."""
        # TODO get from detectron2

        proposals = subsample_labels(labels, boxes, ...)

        return proposals
