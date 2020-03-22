from typing import List, Optional, Dict, Any
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
import functools
sns.set_style("darkgrid")
plt.style.use(["dark_background"])


def plot_fields(df: pd.DataFrame, ys: List[Any], x: Any='iteration', labels: Optional[List[str]]=None,
                preprocessors=[], styles: Dict[str, str]={}, fig: Optional[matplotlib.figure.Figure]=None):
    if labels is None:
        labels = ys

    for preprocessor in preprocessors:
        df = preprocessor(df, [x] + ys)

    if fig is None:
        fig = plt.figure(figsize=(12, 8))
    for y, label in zip(ys, labels):
        plot_df = df.dropna(subset=[x, y])
        x_values = plot_df[x]
        y_values = plot_df[y]
        if y in styles:
            plt.plot(x_values, y_values, styles[y], label=label)
        else:
            plt.plot(x_values, y_values, label=label)
    plt.legend()
    return fig


def plot_hyper_comparison(all_dfs: pd.DataFrame, hyper_param: str, ys: List[str], x: str='iteration', *args, **kwargs):
    compare_df = all_dfs.pivot_table(index=x, columns=[hyper_param], values=ys, aggfunc='first').reset_index()

    compare_df.columns = compare_df.columns.to_flat_index()

    hyper_values = all_dfs[hyper_param].unique()
    return plot_fields(compare_df, x=(x, ''),
                       ys=[(field, hyper_value) for hyper_value in hyper_values for field in ys], *args, **kwargs)


def _smoothen_fields(df, fields, win_size=100):
    #df = df.dropna(subset=fields)[fields]
    df = df.fillna(method='ffill')[fields]
    df = df.rolling(win_size, win_type='parzen').mean()
    return df.dropna()


def smoothen_fields(win_size=100):
    return functools.partial(_smoothen_fields, win_size=win_size)


def subsample(rate):
    return lambda df, _: df.iloc[::rate]


def dropna(subset):
    return lambda df: df.dropna(subset=subset)
