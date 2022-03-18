import pandas as pd
from os import listdir
from os.path import isfile, join
import re

def merge_and_save(files, name):
    print(files)
    df_from_each_file = (pd.read_csv(f, sep=',') for f in files)
    df_merged  = pd.concat(df_from_each_file, ignore_index=True)
    df_merged = df_merged.loc[(df_merged.sum(axis=1) > 1), (df_merged.sum(axis=0) > 1)]
    for c in df_merged.columns:
        if "Counter" not in c:#"Messages-Age" in c or "Max_Consecutive" in c:
            # mx = np.ma.masked_array(df_merged[c], mask=df_merged[c] == 0)
            # minv = mx.min()
            # df_merged[c] = (df_merged[c] - minv)/10e9
            # df_merged[c][df_merged[c] < 0] = 0
            del df_merged[c]
    df_merged.to_csv(name, index=False)


def sort_predicate(value):
    nums = re.findall(r'\d+', value)
    return int(nums[0])

def merge_normal_data():
    MERGED_NAME = './normal/merged_normal_build_count.csv'
    PATH = './normal/building/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

    MERGED_NAME = './normal/merged_normal_cans_count.csv'
    PATH = './normal/cans/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

    MERGED_NAME = './normal/merged_normal_corr_count.csv'
    PATH = './normal/corr/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

def merge_anomaly_laser_fault():
    MERGED_NAME = './test/merged_laser_fault_build_count.csv'
    PATH = './test/laser_fault/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f)) and "build" in f]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

    MERGED_NAME = './test/merged_laser_fault_cans_count.csv'
    PATH = './test/laser_fault/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f)) and "cans" in f]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

    MERGED_NAME = './test/merged_laser_fault_corr_count.csv'
    PATH = './test/laser_fault/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f)) and "corr" in f]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

def merge_anomaly_obs():
    MERGED_NAME = './test/merged_obs_cans.csv'
    PATH = './test/obs/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f)) and "cans" in f]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

    MERGED_NAME = './test/merged_obs_corr.csv'
    PATH = './test/obs/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f)) and "corr" in f]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

def merge_anomaly_software_fault():
    MERGED_NAME = './test/merged_software_fault.csv'
    PATH = './test/software_fault/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)

def merge_anomaly_velocity_attack():
    MERGED_NAME = './test/merged_velocity_attack.csv'
    PATH = './test/velocity_attack/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)


def merge_pick_normal():
    MERGED_NAME = './normal/merged_normal_pick.csv'
    PATH = './normal/pick/1/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)


def merge_pick_anomaly_miss_cup():
    MERGED_NAME = './test/merged_pick_miss_cup.csv'
    PATH = './test/pick/miss_cup/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)


def merge_pick_anomaly_restrict_vision():
    MERGED_NAME = './test/merged_pick_restrict_vision.csv'
    PATH = './test/pick/restricted_vision/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)


def merge_pick_anomaly_stolen():
    MERGED_NAME = './test/merged_pick_stolen.csv'
    PATH = './test/pick/stolen/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)


def merge_pick_anomaly_stuck():
    MERGED_NAME = './test/merged_pick_stuck.csv'
    PATH = './test/pick/stuck/'
    onlyfiles = [PATH + f for f in listdir(PATH) if isfile(join(PATH, f))]
    onlyfiles.sort(key=sort_predicate)
    merge_and_save(onlyfiles, MERGED_NAME)


# merge_pick_normal()
# merge_pick_anomaly_miss_cup()
# merge_pick_anomaly_restrict_vision()
# merge_pick_anomaly_stuck()
# merge_pick_anomaly_stolen()

merge_normal_data()
merge_anomaly_laser_fault()
# merge_anomaly_obs()
# merge_anomaly_software_fault()
# merge_anomaly_velocity_attack()
