import numpy as np
import os
import matplotlib.pyplot as plt
from scipy.stats import ks_2samp

# Consts
INF=0
SUP=200


def load_directory(directory, file_prefix):
    runs = []
    for i in range(len(os.listdir(directory))):
        f = os.path.join(directory, file_prefix + str(i + 1) + ".csv")
        if os.path.isfile(f):
            file_data = np.genfromtxt(f, delimiter=",", skip_header=1)
            runs.append(file_data)
    return runs, np.concatenate([np.reshape(r, -1) for r in runs])


def plot_hist(runs, title):
    fig, axs = plt.subplots(1)
    for i, r in enumerate(runs):
        axs.hist(np.reshape(r[np.logical_and(r > INF, r < SUP)], -1), bins='auto', density=True)
    axs.set_title(title)
    axs.set_xlim((INF, SUP))


def ks_test(runs, more_runs=[]):
    ks_test = []
    p_values = []
    if len(more_runs) == 0:  # Compare to itself
        for j, rj in enumerate(runs):
            other_runs = np.concatenate([np.reshape(runs[i], -1) for i in range(len(runs)) if i != j])
            ks_test.append(ks_2samp(np.reshape(other_runs, -1), np.reshape(rj, -1)).statistic)
            p_values.append(ks_2samp(np.reshape(other_runs, -1), np.reshape(rj, -1)).pvalue)
    else:  # Compare to other
        n = 2
        for j, rj in enumerate(runs):
            rj = rj[len(rj) * (n - 2) // 3:]
            ks_test.append(ks_2samp(np.reshape(more_runs, -1), np.reshape(rj, -1)).statistic)
            p_values.append(ks_2samp(np.reshape(more_runs, -1), np.reshape(rj, -1)).pvalue)
    print(p_values)
    return ks_test


def plot_ks_test_results(norm_norm, norm_anomaly, anomaly_anomaly, title=''):
    plt.title(title)
    plt.hist(norm_norm, bins="auto", density=True, alpha=0.5, label='normal vs normal')
    plt.hist(norm_anomaly, bins="auto", density=True, alpha=0.5, label='anomaly vs normal')
    plt.hist(anomaly_anomaly, bins="auto", density=True, alpha=0.5, label='anomaly vs anomaly')
    plt.legend()


TITLE_NORMAL = 'Pick normal'
TITLE_ANOMALY = 'Pick and restricted vision'
NORM_COUNTS_PATH = './data/normal/pick'
NORM_FILE_PREFIX = 'pick_normal'
ANOMALY_COUNTS_PATH = './data/anomaly/pick/restricted_vision'
ANOM_FILE_PREFIX = 'counts_restricted_vision'

runs, all_runs = load_directory(NORM_COUNTS_PATH, NORM_FILE_PREFIX)
plot_hist(runs, TITLE_NORMAL)

anomaly_runs, all_anomaly_runs = load_directory(ANOMALY_COUNTS_PATH, ANOM_FILE_PREFIX)
plot_hist(anomaly_runs, TITLE_ANOMALY)

ks_normal_normal = ks_test(runs)
ks_normal_anomaly = ks_test(anomaly_runs, all_runs)
ks_anomaly_anomaly = ks_test(anomaly_runs)
plot_ks_test_results(ks_normal_normal, ks_normal_anomaly, ks_anomaly_anomaly, TITLE_ANOMALY)