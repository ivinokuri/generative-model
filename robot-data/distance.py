#!/usr/bin/env python

import argparse
import os
import sys
from argparse import Namespace
from typing import AnyStr, Dict, List
import numpy as np
from numpy import ndarray
from matplotlib import pyplot as plt


def get_args(args=None) -> Namespace:
    if args is None:
        args = sys.argv[1:]

    parser = argparse.ArgumentParser()
    parser.add_argument("-s",
                        "--shift",
                        nargs="+",
                        type=int,
                        default=[1],
                        help="number of shift")
    parser.add_argument("-r",
                        "--shuffle",
                        type=int,
                        default=100,
                        help="number of shuffle")
    parser.add_argument("folder", nargs='+', help="folder to load data from")

    return parser.parse_args(args)


def get_path(args: Namespace) -> Dict[AnyStr, List[AnyStr]]:
    paths = {}
    for folder in args.folder:
        name = os.path.basename(folder)
        for file in os.listdir(os.path.join(folder, 'counts_only')):
            if file.endswith('.csv'):
                paths.setdefault(name, []).append(
                    os.path.join(folder, 'counts_only', file))
    return paths


def load_data(paths: Dict[AnyStr, List[AnyStr]]):
    data = {}
    for name, paths in paths.items():
        for path in paths:
            a = np.genfromtxt(path, delimiter=',', dtype=np.float64)
            data.setdefault(name, []).append(a[1:])

    return data


def distance(a: ndarray, b: ndarray) -> float:
    return (np.linalg.norm(a - b, axis=1) / b.shape[0]).mean()


def random_dis(a: ndarray, shuffle: int) -> float:
    s = 0
    b = a.copy()
    for _ in range(shuffle):
        np.random.shuffle(b)
        s += distance(a, b)
    return s / shuffle


def shift_dis(a: ndarray, shift: int) -> float:
    b = a[shift:]
    a = a[:-shift]
    return distance(a, b)


def table_dis(a: ndarray, args: Namespace):
    shifts = []
    for shift in args.shift:
        shifts.append(shift_dis(a.copy(), shift))
    shuffle = random_dis(a.copy(), args.shuffle)
    return shifts, shuffle


def main(args: Namespace):
    args = get_args()
    paths = get_path(args)
    data = load_data(paths)
    dis = {}
    for name, arrays in data.items():
        dis[name] = {'shuffle': []}
        print(name)
        for a in arrays:
            sht, she = table_dis(a, args)
            for idx, shift in enumerate(args.shift):
                dis[name].setdefault(f'shift-{shift}', []).append(sht[idx])
            dis[name]['shuffle'].append(she)

    boxplot_dis(dis)


def boxplot_dis(dis: Dict[AnyStr, Dict[AnyStr, List[float]]]):
    n = len(dis)
    fig, axs = plt.subplots(1, n, figsize=(n * 4, 4))
    fig.suptitle('Distance')
    for ax, (name, d) in zip(axs, dis.items()):
        ax.set_title(name)
        ax.boxplot(d.values())
        ax.set_xticks(ticks=list(range(1, len(d) + 1)), labels=d.keys())
    plt.show()


if __name__ == '__main__':
    main(get_args())
