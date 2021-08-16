"""VisT module registry."""

from abc import ABCMeta
from typing import Any, Dict, Optional, Tuple


class RegistryHolder(type):
    """Registry for all modules in VisT."""

    REGISTRY: Dict[str, "RegistryHolder"] = {}

    # Ignore mcs vs. cls since it conflicts with PEP8:
    # https://github.com/PyCQA/pylint/issues/2028
    def __new__(  # type: ignore # pylint: disable=bad-mcs-classmethod-argument
        cls, name: str, bases: Tuple[Any], attrs: Dict[str, Any]
    ) -> "RegistryHolder":
        """Method called when constructing a new class.

        Adds the new class to the detect registry,
        with its module + name as key.
        """
        new_cls = type.__new__(cls, name, bases, attrs)
        assert isinstance(new_cls, RegistryHolder)
        if len(bases):  # must inherit from some base class beyond Registry
            base = str(bases[0]).replace("<class '", "").replace("'>", "")
            module_name = ".".join([*base.split(".")[:-2], new_cls.__name__])
            cls.REGISTRY[module_name] = new_cls
        return new_cls

    @classmethod
    def get_registry(  # pylint: disable=bad-mcs-classmethod-argument
        cls, cls_type: Optional["RegistryHolder"] = None
    ) -> Dict[str, "RegistryHolder"]:
        """Get registered classes, optionally for a specific scope.

        Args:
            cls_type: The super class for which you'd like to get the
            registered subclasses. E.g. input vist.model.BaseModel to get
            all registered models.

        Returns:
            Dict[str, RegistryHolder]: A dictionary with class names as keys
            and classes as values.
        """
        if cls_type is not None:
            return {
                k.split(".")[-1]: v
                for k, v in cls.REGISTRY.items()
                if issubclass(v, cls_type)
            }

        return dict(cls.REGISTRY)  # pragma: no cover


class ABCRegistryHolder(RegistryHolder, ABCMeta):
    pass