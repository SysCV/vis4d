"""Vis4D module registry."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover
    from vis4d.struct import DictStrAny


class RegistryHolder(type):
    """Registry for all modules in Vis4D."""

    REGISTRY: Dict[str, "RegistryHolder"] = {}

    # Ignore mcs vs. cls since it conflicts with PEP8:
    # https://github.com/PyCQA/pylint/issues/2028
    def __new__(  # type: ignore # pylint: disable=bad-mcs-classmethod-argument
        cls, name: str, bases: Tuple[Any], attrs: DictStrAny
    ) -> "RegistryHolder":
        """Method called when constructing a new class.

        Adds the new class to the detect registry,
        with its module + name as key.
        """
        new_cls = type.__new__(cls, name, bases, attrs)
        assert isinstance(new_cls, RegistryHolder)
        if len(bases):  # inherits from some base class beyond Registry
            base_name = bases[0]
        else:
            base_name = str(new_cls)

        base = str(base_name).replace("<class '", "").replace("'>", "")
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
            registered subclasses. E.g. input vis4d.model.BaseModel to get
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
