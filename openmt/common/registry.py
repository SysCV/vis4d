"""OpenMT module registry."""
from typing import Any, Dict, Optional, Tuple


class RegistryHolder(type):
    """Registry for all modules in openMT."""

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
        module_name = ".".join(
            [*attrs["__module__"].split(".")[:-1], new_cls.__name__]
        )
        cls.REGISTRY[module_name] = new_cls
        return new_cls

    @classmethod
    def get_registry(  # pylint: disable=bad-mcs-classmethod-argument
        cls, scope: Optional[str] = None
    ) -> Dict[str, "RegistryHolder"]:
        """Get registered classes, optionally for a specific scope.

        Args:
            scope: indicates module to pull classes from,
            e.g.  'module.submodule' will return all registered classes in
        'submodule'.

        Returns:
            Dict[str, Type]: A dictionary with class names as keys and
            classes as values.
        """
        if scope is not None:
            return {
                k.replace(scope + ".", ""): v
                for k, v in cls.REGISTRY.items()
                if k.startswith(scope)
            }

        return dict(cls.REGISTRY)  # pragma: no cover
