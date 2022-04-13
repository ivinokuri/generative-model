#!/usr/bin/env python3
import os.path

import pandas as pd
from utils import list_files

def only_count_data(source_dir, dest_dir):
    PATH = source_dir
    if not os.path.isdir(dest_dir):
        os.mkdir(dest_dir)
    onlyfiles = list_files(PATH)
    for f in onlyfiles:
        _, tail = os.path.split(f)
        data = pd.read_csv(f, sep=',')
        for c in data.columns:
            if "Counter" not in c:
                del data[c]
        print("processing " + tail + ' ' + str(len(data.columns)))
        data.to_csv(dest_dir + 'counts_' + tail, index=False)

def main():
    # pick normal
    only_count_data('../../../robot-data/new_data/normal/pick/1/',
                    '../../../robot-data/new_data/normal/pick/counts_only/')
    # corr normal
    only_count_data('../../../robot-data/new_data/normal/corr/',
                    '../../../robot-data/new_data/normal/corr/counts_only/')
    # corr normal
    only_count_data('../../../robot-data/new_data/normal/cans/',
                    '../../../robot-data/new_data/normal/cans/counts_only/')
    # building normal
    only_count_data('../../../robot-data/new_data/normal/building/',
                    '../../../robot-data/new_data/normal/building/counts_only/')

    # laser fault build
    only_count_data('../../../robot-data/new_data/test/laser_fault/build/',
                    '../../../robot-data/new_data/test/laser_fault/build/counts_only/')
    # laser fault cans
    only_count_data('../../../robot-data/new_data/test/laser_fault/cans/',
                    '../../../robot-data/new_data/test/laser_fault/cans/counts_only/')
    # laser fault corr
    only_count_data('../../../robot-data/new_data/test/laser_fault/corr/',
                    '../../../robot-data/new_data/test/laser_fault/corr/counts_only/')
    # obs build
    only_count_data('../../../robot-data/new_data/test/obs/cans/',
                    '../../../robot-data/new_data/test/obs/cans/counts_only/')
    # obs build
    only_count_data('../../../robot-data/new_data/test/obs/corr/',
                    '../../../robot-data/new_data/test/obs/corr/counts_only/')
    # pick miss cup test
    only_count_data('../../../robot-data/new_data/test/pick/miss_cup/',
                    '../../../robot-data/new_data/test/pick/miss_cup/counts_only/')
    # pick restricted vision test
    only_count_data('../../../robot-data/new_data/test/pick/restricted_vision/',
                    '../../../robot-data/new_data/test/pick/restricted_vision/counts_only/')
    # pick stolen test
    only_count_data('../../../robot-data/new_data/test/pick/stolen/',
                    '../../../robot-data/new_data/test/pick/stolen/counts_only/')
    # pick stuck test
    only_count_data('../../../robot-data/new_data/test/pick/stuck/',
                    '../../../robot-data/new_data/test/pick/stuck/counts_only/')
    # software fault test
    only_count_data('../../../robot-data/new_data/test/software_fault/',
                    '../../../robot-data/new_data/test/software_fault/counts_only/')
    # velocity attack test
    only_count_data('../../../robot-data/new_data/test/velocity_attack/',
                    '../../../robot-data/new_data/test/velocity_attack/counts_only/')

if __name__ == '__main__':
    main()