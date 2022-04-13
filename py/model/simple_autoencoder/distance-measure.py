#!/usr/bin/env python3

from utils import get_file_paths, list_files
import numpy as np
import pandas
import matplotlib.pylab as plt

from sklearn.preprocessing import StandardScaler

def main():
    normal_dir_paths, anomaly_dir_paths = get_file_paths()
    for o in normal_dir_paths:
        key = list(o.keys())[0]
        value = list(o.values())[0]
        data_files = list_files(value)
        for f in data_files:
            file_data = pandas.read_csv(f)
            file_data = normalize_data(file_data)
            distances = []
            for i in range(len(file_data.index) - 1):
                distances.append(np.linalg.norm(file_data.values[i] - file_data.values[i + 1]))
            fig, ax = plt.subplots()
            ax.hist(distances, bins=200, density=True, stacked=True, label="Seq")
            ax.legend()
            ax.set_title('Mean ' + str(np.mean(distances)) + ' interval ' + str(np.percentile(distances, [90])))
            plt.legend()
            plt.show()
            rand_distances = np.zeros(len(file_data.index) - 1)
            for _ in range(100):
                rd = []
                for i in range(len(file_data.index) - 1):
                    rand_index = np.random.randint(0, len(file_data.index))
                    rd.append(np.linalg.norm(file_data.values[i] - file_data.values[rand_index]))
                rand_distances += rd
            rand_distances /= 100
            fig, ax = plt.subplots()
            ax.hist(rand_distances, bins=200, density=True, stacked=True, label="Random")
            ax.set_title('Mean ' + str(np.mean(rand_distances)) + ' interval ' + str(np.percentile(rand_distances, [90])))
            ax.legend()
            plt.legend()
            plt.show()

            denom = 0
            for i in range(10000):
                rand_1 = np.random.randint(0, len(distances))
                rand_2 = np.random.randint(0, len(rand_distances))
                if distances[rand_1] < rand_distances[rand_2]:
                    denom += 1

            print('Probability ' + str(denom/10000.0))

            return 1

scalers = {}
def normalize_data(data, name=""):
        norm_data = data
        for i in data.columns:
            scaler = None
            if (name + '_scaler_' + i) not in scalers:
                scaler = StandardScaler()
            else:
                scaler = scalers[name + '_scaler_' + i]
            s_s = scaler.fit_transform(norm_data[i].values.reshape(-1, 1))
            s_s = np.reshape(s_s, len(s_s))
            scalers[name + '_scaler_' + i] = scaler
            norm_data[i] = s_s
        return norm_data

if __name__ == '__main__':
    main()