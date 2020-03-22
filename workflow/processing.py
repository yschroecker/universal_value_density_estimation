from typing import Sequence

import sqlite3
import pandas as pd
import os
import json


def get_run_by_id_file(observer_path, index):
    path = os.path.join(observer_path, str(index))
    run_file = os.path.join(path, 'run.json')
    with open(run_file, 'r') as f:
        run = json.load(f)
    info_file = os.path.join(path, 'info.json')
    with open(info_file, 'r') as f:
        info = json.load(f)
    run['info'] = info
    config_file = os.path.join(path, 'config.json')
    with open(config_file, 'r') as f:
        config = json.load(f)
    run['config'] = config
    run['_id'] = index
    return run


class Experiment:
    def __init__(self, experiment_id: int, sacred_path: str):
        self._run = get_run_by_id_file(sacred_path, experiment_id)
        self.xid = experiment_id

    def record(self):
        db_path = self._run['info']['records']
        conn = sqlite3.connect(db_path)
        df = pd.read_sql_query("select * from records", conn)
        if len(df) > 0:
            df['minute'] = (df['timestamp'] - df.loc[0, 'timestamp'])/60
        return df

    def config(self):
        items = list(self._run['config'].items())
        final_config = {}
        while len(items) > 0:
            key, value = items[0]
            if isinstance(value, dict):
                for nested_key, nested_value in value.items():
                    items.append((f'{key}.{nested_key}', nested_value))
            else:
                final_config[key] = value
            del items[0]
        return pd.Series(final_config)

    def __repr__(self):
        return self._run['experiment']['name']


def load_combined_record(experiments: Sequence[Experiment]):
    dfs = [experiment.record() for experiment in experiments]
    configs = [ex.config() for ex in experiments]
    config_df = pd.DataFrame(configs)
    relevant_hypers = [column for column in config_df.columns if config_df[column].nunique() > 1 and column != 'seed']
    for i, df in enumerate(dfs):
        for relevant_hyper in relevant_hypers:
            df[relevant_hyper] = config_df.loc[i, relevant_hyper]
        df['xid'] = experiments[i].xid
    return pd.concat(dfs)


def _test():
    from workflow import plotting
    experiment = Experiment(855, "/home/anon/generated_data/sacred")
    df = experiment.record()
    fig = plotting.plot_fields(df, ['MADE_MADE_loss'])
    fig.show()
    print()


if __name__ == '__main__':
    _test()

