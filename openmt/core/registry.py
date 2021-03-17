"""OpenMT module registry."""
from typing import Dict, Optional, Type


class RegistryHolder(type):

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        """When constructing a new class, add the new class to the model
        registry, with its module + name as key."""
        new_cls = type.__new__(cls, name, bases, attrs)
        module_name = ".".join(
            [*attrs["__module__"].split(".")[:-1], new_cls.__name__]
        )
        cls.REGISTRY[module_name] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls, scope: Optional[str] = None) -> Dict[str, Type]:
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
        else:
            return dict(cls.REGISTRY)
