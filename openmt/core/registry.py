"""OpenMT module registry."""


class RegistryHolder(type):

    REGISTRY = {}

    def __new__(cls, name, bases, attrs):
        """When constructing a new class, add the new class to the model
        registry, with its module + name as key."""
        new_cls = type.__new__(cls, name, bases, attrs)
        module_name = ".".join(
            [*attrs["__module__"].split(".")[1:-1], new_cls.__name__]
        )
        cls.REGISTRY[module_name] = new_cls
        return new_cls

    @classmethod
    def get_registry(cls, scope=None):
        """Get registry, optionally for specific scope."""
        if scope is not None:
            return {
                k.replace(scope, ""): v
                for k, v in cls.REGISTRY.items()
                if k.startswith(scope)
            }
        else:
            return dict(cls.REGISTRY)
