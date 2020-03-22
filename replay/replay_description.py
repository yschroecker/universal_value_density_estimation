from typing import Sequence, NamedTuple, Optional, Dict, Union, Type
import numpy as np


class ReplayColumn(NamedTuple):
    name: str
    width: int
    default: Optional[float] = None


class ReplayDescription:
    def __init__(self, description: Sequence[ReplayColumn], named_tuple_type: Optional[Type[NamedTuple]]=None):
        self._description = []
        self._total_width = 0
        self._indices = {}
        for column in description:
            if column.width == 0:
                width = 1
                slice_ = self._total_width
                self._indices[column.name] = slice_
            else:
                width = column.width
                slice_ = slice(self._total_width, self._total_width + column.width)
                self._indices[column.name] = slice_.start
            self._description.append((column.name, slice_, column.default))
            self._total_width += width
        self._named_tuple_type = named_tuple_type

    def prepare_samples(self, shape: Sequence[int], samples: Union[Dict[str, np.ndarray], NamedTuple]) -> np.ndarray:
        if isinstance(samples, NamedTuple):
            samples = samples._asdict()
        buffer_sample = np.zeros((*shape, self._total_width))
        for field, column_slice, default_val in self._description:
            if field in samples:
                buffer_sample[..., column_slice] = samples[field]
            elif default_val is not None:
                buffer_sample[..., column_slice] = default_val
        return buffer_sample

    def parse_sample(self, samples: np.ndarray) -> Union[Dict[str, np.ndarray], NamedTuple]:
        result = {}
        for field, column_slice, _ in self._description:
            result[field] = samples[..., column_slice]
        if self._named_tuple_type is not None:
            result = self._named_tuple_type(**{k: result[k] for k in self._named_tuple_type._fields})
        return result

    def get_index(self, field: str):
        return self._indices[field]

    @property
    def num_columns(self) -> int:
        return self._total_width


def _test():
    description = ReplayDescription([ReplayColumn("x", 2), ReplayColumn("y", 2), ReplayColumn("z", 3)])
    samples = {"x": np.ones((10, 2, 2)), "z": 1.5 * np.ones((10, 2, 3))}
    samples = description.prepare_samples((10, 2), samples)
    print(samples.shape)
    parsed = description.parse_sample(samples)
    print(parsed)


if __name__ == '__main__':
    _test()

