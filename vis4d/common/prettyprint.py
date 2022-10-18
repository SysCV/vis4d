"""Utilities for pretty printing."""


class PrettyRepMixin:
    """Creates a pretty string representation of a class with parameters."""

    def __repr__(self) -> str:
        """Print class & params, s.t. user can inspect easily via cmd line."""
        attr_str = ""
        for k, v in vars(self).items():
            if k != "type" and not k.startswith("_"):
                attr_str += f"{k}={str(v)}, "
        attr_str = attr_str.rstrip(", ")
        return f"{self.__class__.__name__}({attr_str})"
