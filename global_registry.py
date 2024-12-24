
# config object for yaml
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Registry:
    def __init__(self):
        self._registry = {}

    def set(self, key, value):
        self._registry[key] = value

    def get(self, key, alt=None):
        return self._registry.get(key, alt)
    
    def update_from_config(self, config):
        if isinstance(config, Config):
            def recurse_update(prefix, obj):
                for key, value in obj.__dict__.items():
                    full_key = f"{prefix}.{key}" if prefix else key
                    if isinstance(value, Config):
                        recurse_update(full_key, value)
                    else:
                        self.set(full_key, value)
            recurse_update("", config)
        else:
            raise TypeError("Expected a Config object")


# global registry object
PARAM_REGISTRY = Registry()