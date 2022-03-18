#!/usr/bin/env python3
import os.path

import pandas as pd
from utils import list_files

def only_count_data(source_dir, dest_dir):
    PATH = source_dir
    onlyfiles = list_files(PATH)
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