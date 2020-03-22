from typing import Type, NamedTuple
import sacred.config


class ConfigParser:
    def __init__(self, named_tuple: Type[NamedTuple]):
        self._named_tuple = named_tuple

    def parse(self, config: sacred.config.ConfigDict):
        # noinspection PyProtectedMember
        filtered_config = {k: config[k] for k in self._named_tuple._fields if k in config}
        return self._named_tuple(**filtered_config)