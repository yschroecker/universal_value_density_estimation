from typing import Union, Sequence
import abc
import sqlite3
import sacred
import datetime
import os
import uuid
import numpy as np
import time


class ReporterBase(metaclass=abc.ABCMeta):
    def register_field(self, field: str):
        pass

    def finalize_fields(self):
        pass

    def iter_record(self, name: str, value: Union[np.ndarray, float]):
        pass

    def iterate(self, num_iterations: int=1):
        pass


class NoReporter(ReporterBase):
    pass


class SimpleReporter(ReporterBase):
    def __init__(self, exponential: float):
        self._exp = exponential
        self._fields = {}

    def register_field(self, field: str):
        pass

    def finalize_fields(self):
        pass

    def iter_record(self, name: str, value: Union[np.ndarray, float]):
        if name not in self._fields:
            self._fields[name] = value
        self._fields[name] = self._fields[name] * self._exp + value * (1 - self._exp)

    def iterate(self, num_iterations: int=1):
        pass

    def get_description(self, names: Sequence[str]):
        return ", ".join([f"{name}: {self._fields.get(name, np.nan)}" for name in names])


class Reporter(SimpleReporter):
    def __init__(self, experiment: sacred.Experiment, storage_dir: str,
                 store_frequency: datetime.timedelta=datetime.timedelta(seconds=15),
                 table_name: str='records',
                 exponetial_average: float=0.99):
        super().__init__(exponetial_average)
        self._ex = experiment
        self._iteration = 0
        if not os.path.exists(storage_dir):
            os.makedirs(storage_dir)
        self._storage_dir = storage_dir
        self._store_path = os.path.join(storage_dir, str(uuid.uuid4()))
        self._ex.info['records'] = self._store_path
        self._last_store = datetime.datetime.now()
        self._store_frequency = store_frequency
        self._connection = sqlite3.connect(self._store_path, timeout=120)
        self._cursor = self._connection.cursor()
        self._table_name = table_name

        self._insert_query = None
        self._columns = []
        self._current_row = {}

    def sub_reporter(self, table_name: str):
        assert table_name != self._table_name
        return Reporter(self._ex, self._storage_dir, self._store_frequency, table_name)

    def register_field(self, field: str):
        super().register_field(field)
        self._columns.append(field)
        return field

    def finalize_fields(self):
        super().finalize_fields()
        self._cursor.execute(
            f"""create table {self._table_name} (iteration integer, timestamp integer,
                                      {','.join([column + ' real' for column in self._columns])})""")
        self._connection.commit()
        self._insert_query = f"insert into records values({','.join(['?'] * (len(self._columns) + 2))})"

    def iter_record(self, name: str, value: Union[np.ndarray, float]):
        super().iter_record(name, value)
        if isinstance(value, np.ndarray):
            value = value.item()

        self._current_row[name] = value

    def iterate(self, num_iterations: int=1):
        assert self._insert_query is not None, "you must call finalize_fields"
        super().iterate()
        row = [self._iteration, time.time()] + \
              [self._current_row[col] if col in self._current_row else float('NaN')
               for i, col in enumerate(self._columns)]
        self._current_row = {}
        self._cursor.execute(self._insert_query, row)

        if datetime.datetime.now() - self._last_store > self._store_frequency:
            self._connection.commit()
            self._last_store = datetime.datetime.now()

        self._iteration += num_iterations


_experiment = sacred.Experiment("Test reporter")


@_experiment.automain
def _evaluate_speed():
    reporter = Reporter(_experiment, "/tmp/test.sql")
    reporter.register_field("loss 1")
    reporter.register_field("loss 2")
    reporter.register_field("loss 3")
    reporter.register_field("loss 4")

    import tqdm
    for _ in tqdm.trange(10000000):
        reporter.iter_record("loss 1", 23)
        reporter.iter_record("loss 2", 12)
        reporter.iter_record("loss 3", 17)


