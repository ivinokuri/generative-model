import pandas as pd
import numpy as np
from os import listdir
from os.path import isfile, join

PATH = '../robot-data/thesis_print-master/available runs/navigation scenarios/move_base fault/normal/type'
onlyfiles = [PATH + '1/' + f for f in listdir(PATH + '1') if isfile(join(PATH + '1', f))]
onlyfiles += [PATH + '2/' + f for f in listdir(PATH + '2') if isfile(join(PATH + '2', f))]
onlyfiles += [PATH + '3/' + f for f in listdir(PATH + '3') if isfile(join(PATH + '3', f))]
onlyfiles += [PATH + '4/' + f for f in listdir(PATH + '4') if isfile(join(PATH + '4', f))]

# PATH = '../robot-data/thesis_print-master/available runs/navigation scenarios/move_base fault/test'
# onlyfiles = [PATH + '/' + f for f in listdir(PATH) if isfile(join(PATH, f))]

print(onlyfiles)

df_from_each_file = (pd.read_csv(f, sep=',') for f in onlyfiles)
df_merged  = pd.concat(df_from_each_file, ignore_index=True)
# df_merged.drop(df_merged.columns[df_merged.apply(lambda col: col.sum() == 0)], axis=1)
df_merged = df_merged.loc[(df_merged.sum(axis=1) > 1), (df_merged.sum(axis=0) > 1)]
for c in df_merged.columns:
    if "Messages-Age" in c:
        # mx = np.ma.masked_array(df_merged[c], mask=df_merged[c] == 0)
        # minv = mx.min()
        # df_merged[c] = (df_merged[c] - minv)/10e9
        # df_merged[c][df_merged[c] < 0] = 0
        del df_merged[c]
df_merged.to_csv( "merged.csv")