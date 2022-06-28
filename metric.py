__author__ = "Hao Bian"

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def make_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config', default=None, type=str)
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    # config args
    args = make_parse()

    # file address
    log_path = 'logs/'
    log_name = Path(args.config).parts[-2]
    version_name = Path(args.config).name[:-5]
    log_path = Path(log_path) / log_name / version_name
    result_list = list(log_path.glob('*/result.csv'))

    # save result_list
    df = pd.read_csv(result_list[0], index_col=0)
    metric_name = df.columns.values
    metric = {}
    for i in metric_name:
        metric[i] = []

    for result_dir in result_list:
        result = pd.read_csv(result_dir, index_col=0)
        for i in metric_name:
            metric[i].append(result.loc[0, i])

    metric_output = {}
    for k, v in metric.items():
        if 'test' in k:
            k = k.split('_')[1]
        metric_output[k] = str(round(np.mean(v), 4)) + \
            '±' + str(round(np.std(v), 4))

    for keys, values in metric_output.items():
        print(f'{keys} = {values}')
    print()

    result_mean = pd.DataFrame([metric_output])
    result_mean.to_csv(log_path / 'result_all.csv')
