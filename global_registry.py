
# config object for yaml
class Config:
    def __init__(self, **entries):
        self.__dict__.update(entries)


class Registry:
    def __init__(self):
        self._registry = {}

    def set(self, key, value):
        self._registry[key] = value

    def get(self, key):
        return self._registry.get(key, None)
    
    def update_from_config(self, config):
        if isinstance(config, Config):
            for key, value in config.__dict__.items():
                self.set(key, value)
        else:
            raise TypeError("Expected a Config object")


# global registry object
PARAM_REGISTRY = Registry()
