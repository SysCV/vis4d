"""RoIHead interface for backend."""

import abc


class BaseRoIHead(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(
        self,
    ):
        """Process proposals and output predictions and possibly target
        assignments."""
        raise NotImplementedError
