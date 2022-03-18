#!/usr/bin/env python3
import os.path

import pandas as pd
from os import listdir
from os.path import isfile, join
import re

def sort_predicate(value):
    nums = re.findall(r'\d+', value)
    return int(nums[0])


def only_count_data(source_dir, dest_dir):
    PATH = source_dir
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    for f in onlyfiles:
        _, tail = os.path.split(f)
        data = pd.read_csv(f, sep=',')
        print("processing " + tail)
        for c in data.columns:
            if "Counter" not in c:
                del data[c]
        data.to_csv(dest_dir + 'counts_' + tail, index=False)


def main():
    only_count_data('../../../robot-data/new_data/normal/pick/1/',
                    '../../../robot-data/new_data/normal/pick/counts_only/')
    only_count_data('../../../robot-data/new_data/test/pick/miss_cup/',
                    '../../../robot-data/new_data/test/pick/miss_cup_counts/')

if __name__ == '__main__':
    main()