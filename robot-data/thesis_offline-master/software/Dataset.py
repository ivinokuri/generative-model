from collections import namedtuple
import numpy as np
from Style import Configure as Conf
ROWS_INDEX = 0
COLOMUNS_INDEX = 1

DataPred = namedtuple("DataPred", "paths datasets predictions")


class Datasets:
    # trainings_dir_path
    # def __init__(self, positives_dir_paths, negatives_dir_path, trainings_dir_path, test_size=0.5, delete=10):
    def __init__(self, positives_dir_paths, negatives_dir_path, test_size=0.5, delete=10):
        if negatives_dir_path:
            positives_dir_paths = Datasets.get_dirs_names(positives_dir_paths)
        self.test_size = test_size
        positives_paths = []
        trainings_paths = []
        if negatives_dir_path:
            for pos_dir_path in positives_dir_paths:
                files = Datasets.get_files_name(pos_dir_path)
                trainings_files, positives_files = Datasets.__test_split(files, test_size)
                # positives_paths.extend(files)
                positives_paths.extend(positives_files)
                trainings_paths.extend(trainings_files)
        else:
            files = Datasets.get_files_name(positives_dir_paths)
            trainings_files, positives_files = Datasets.__test_split(files, test_size)
            positives_paths.extend(positives_files)
            trainings_paths.extend(trainings_files)

        if not positives_paths:
            print "positives_paths: empty"

        if not trainings_paths:
            print "trainings_paths: empty"

        # files = Datasets.get_files_name(trainings_dir_path)
        # trainings_paths.extend(files)
        negatives_paths = []
        # for neg_dir_path in negatives_dir_path:
        if negatives_dir_path:
            files = Datasets.get_files_name(negatives_dir_path)
            negatives_paths.extend(files)
        if not negatives_paths:
            print "negatives_paths: empty"

        trainings, positives, negatives = Datasets.read_csv_from_paths(trainings_paths, positives_paths, negatives_paths)
        trainings, positives, negatives = Datasets.remove_rows_from(delete, trainings, positives, negatives)

        self.trainings_preds = DataPred(trainings_paths, trainings, self.__create_predictions(trainings))
        self.positives_preds = DataPred(positives_paths, positives, self.__create_predictions(positives))
        self.negatives_preds = DataPred(negatives_paths, negatives, self.__create_predictions(negatives))

    @staticmethod
    def get_dirs_names(path):
        import glob
        files = glob.glob(path + "*")
        return [f + "/" for f in files]

    @staticmethod
    def remove_rows_from(delete, *datasets_types):
        def remove_from(df):
            return df[delete:]# -delete]
        if delete == 0:
            return datasets_types
        ret = []
        for dfs in datasets_types:
            ret.append(map(lambda df: remove_from(df), dfs))
        return ret

    def get_predictions(self):
        trainings_preds = self.trainings_preds.predictions
        positives_preds = self.positives_preds.predictions
        negatives_preds = self.negatives_preds.predictions
        return trainings_preds, positives_preds, negatives_preds

    def get_names(self):
        trainings_names = map(lambda p: Datasets.__get_filename_from_path(p), self.trainings_preds.paths)
        positives_names = map(lambda p: Datasets.__get_filename_from_path(p), self.positives_preds.paths)
        negatives_names = map(lambda p: Datasets.__get_filename_from_path(p), self.negatives_preds.paths)
        return trainings_names, positives_names, negatives_names

    def update_predictions(self, trainings_preds, positives_preds, negatives_preds):
        self.trainings_preds = DataPred(self.trainings_preds.paths, self.trainings_preds.datasets, map(lambda x,y: self.intersection_labeling(x,y), trainings_preds, self.trainings_preds.predictions))
        self.positives_preds = DataPred(self.positives_preds.paths, self.positives_preds.datasets, map(lambda x,y: self.intersection_labeling(x,y), positives_preds, self.positives_preds.predictions))
        if negatives_preds:
            self.negatives_preds = DataPred(self.negatives_preds.paths, self.negatives_preds.datasets, map(lambda x,y: self.intersection_labeling(x,y), negatives_preds, self.negatives_preds.predictions))

    def get_datasets(self):
        trainings = self.trainings_preds.datasets
        positives = self.positives_preds.datasets
        negatives = self.negatives_preds.datasets
        return trainings, positives, negatives

    def set_datasets(self,training_datasets, positive_datasets, negative_datasets):
        self.trainings_preds = DataPred(self.trainings_preds.paths, training_datasets, self.trainings_preds.predictions)
        self.positives_preds = DataPred(self.positives_preds.paths, positive_datasets, self.positives_preds.predictions)
        if negative_datasets:
            self.negatives_preds = DataPred(self.negatives_preds.paths, negative_datasets, self.negatives_preds.predictions)

    def __create_predictions(self, datasets):
        ret = []
        for df in datasets:
            df_len = df.shape[ROWS_INDEX]
            preds = [Conf.POSITIVE_LABEL] * df_len
            ret.append(preds)
        return ret

    def intersection_labeling(self, labeling, *labelings):
        new_labeling = []
        labelings = list(labelings)
        labelings.append(labeling)
        for i in range(len(labeling)):
            lbl = True
            for labeling in labelings:
                lbl = lbl and (labeling[i] == Conf.POSITIVE_LABEL)
            new_labeling.append(Conf.POSITIVE_LABEL if lbl else Conf.NEGATIVE_LABEL)
        # print new_labeling
        return new_labeling

    @staticmethod
    def get_files_name(path):
        import glob
        csv_suffix = ".csv"
        if path[-len(csv_suffix):] == csv_suffix:
            files = [path]
        else:
            files = glob.glob(path + ("*" + csv_suffix))
        return files

    @staticmethod
    def read_csv_from_paths(*csvs_paths):
        import pandas as pd
        ret = []
        for csvs_path in csvs_paths:
            dfs = map(lambda x: pd.read_csv(x, header=0), csvs_path)
            ret.append(dfs)
        return ret

    @staticmethod
    def __test_split(paths, _test_size):
        from sklearn.model_selection import train_test_split
        train, test = train_test_split(paths, test_size=_test_size)
        return train, test

    @staticmethod
    def __get_filename_from_path(path):
        return path.split('/')[-1]

    def __str__(self):
        def get_dir_path(path):
            return '/'.join(path.split('/')[:-1])
        def count_values(val, *preds):
            count = 0
            for pred in preds:
                count += (np.array(pred) == val).sum()
            return count
        def create_info(title, paths, results):
            if len(paths) < 1:
                return ""
            directory = get_dir_path(paths[0])
            information = "%s directory: %s\n" % (title, directory)
            # information += "%s files: %s\n" % (title, str(map(Datasets.__get_filename_from_path, paths)))
            pos_count = count_values(Conf.POSITIVE_LABEL, *results)
            neg_count = count_values(Conf.NEGATIVE_LABEL, *results)
            information += "positive predictions: %s/%s, negative predictions: %s/%s\n" % (pos_count, pos_count+neg_count, neg_count, pos_count+neg_count)
            return information
        trainings_preds, positives_preds, negatives_preds = self.get_predictions()
        trainings_location = create_info("trainings", self.trainings_preds.paths, trainings_preds)
        positives_location = create_info("positives", self.positives_preds.paths, positives_preds)
        negatives_location = create_info("negatives", self.negatives_preds.paths, negatives_preds)

        return "%s\n%s\n%s" % (trainings_location, positives_location, negatives_location)

    @staticmethod
    def remove_columns(dataset, columns):
        return dataset.drop(columns, COLOMUNS_INDEX)

    # def __split_array(self, array, indexes_1, indexes_2):
    # array_1 = []
    # array_2 = []
    # for i in indexes_1:
    # array_1.append(array[i])
    # for i in indexes_2:
    # array_2.append(array[i])
    # return array_1, array_2

    # def apply_train_test_split(self, method, _test_size = 0.5):
    # from sklearn.model_selection import train_test_split
    # algorithm = self.__get_machine_learning_method(method)
    # training_sets, positive_sets = train_test_split(self.positive_sets, test_size=_test_size)
    ##print self.positive_sets
    # training_set = pd.concat(training_sets)
    # training_predictions, positive_predictions, negative_predictions = algorithm(training_set, training_sets, positive_sets, self.negative_sets)
    # return training_predictions, positive_predictions, negative_predictions

    # def apply_k_cross_validation(self, method, _n_splits = 3, _shuffle = True):
    # from sklearn.model_selection import KFold
    # algorithm = self.__get_machine_learning_method(method)
    # kf = KFold(n_splits = _n_splits, shuffle = _shuffle)
    # generator = kf.split(self.positive_sets)
    # training_index, testing_index = generator.next()
    # training_sets, positive_sets = self.__split_array(self.positive_sets, training_index, testing_index)
    # training_set = pd.concat(training_sets)
    # training_predictions, positive_predictions, negative_predictions = algorithm(training_set, training_sets, positive_sets, self.negative_sets)
    # return training_predictions, positive_predictions, negative_predictions

    # def set_datasets_by_k_cross_validation(self, positives, negatives _n_splits = 3, _shuffle = True):
    # from sklearn.model_selection import KFold
    # kf = KFold(n_splits = _n_splits, shuffle = _shuffle)
    # generator = kf.split(self.positive_sets)
    # training_index, testing_index = generator.next()
    # new_trainings, new_positives = self.__split_array(positives, training_index, testing_index)
    # self.trainings = new_trainings
    # self.positives = new_positives
    # self.negatives = negatives
    # return new_trainings, new_positives, negatives

    # def set_datasets_by_test_split(self, positives, negatives, _test_size = 0.5, _shuffle = True):
    # from sklearn.model_selection import train_test_split
    # new_trainings, new_positives = train_test_split(positives, test_size=_test_size, shuffle = _shuffle)
    # self.trainings = new_trainings
    # self.positives = new_positives
    # self.negatives = negatives
    # return new_trainings, new_positives, negatives
