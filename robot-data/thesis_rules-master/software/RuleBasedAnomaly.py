import pandas as pd
import numpy
from Style import Style, Configure

ROWS_INDEX = 0
COLOMUNS_INDEX = 1
from Style import Configure as Conf
class NegPos:

    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def update(self, pos, neg):
        self.pos += pos
        self.neg += neg

    def __str__(self):
        return "positives = %s, negatives = %s" % (self.pos, self.neg)

class RuleBasedAnomaly:

    def __init__(self):
        self.training = None
        self.cols_digits = None
        self.pos_columns = None
        self.neg_columns = None
        self.not_neg_columns = None
        self.one_value_columns = None
        self.cov_percen_columns = None
        self.corr_columns = None
        self.min_max_values = None
        self.methods_stats = {Configure.TRAINING:{}, Configure.POSITIVE:{}, Configure.NEGATIVE:{}}
        # self.rules_stats = {} # Used to save learning rules data
        self.deleted_topics = {} # Topics x rules that should not be used during online learning since they were found faulty during offline learning
        self.percentage = None
        self.coverage = None
        self.rules_fit = {}
        self.count = 0

    def fit(self, trainings, percentage, coverage):

        self.percentage = percentage
        self.coverage = coverage
        self.training = pd.concat(trainings, ignore_index=True)

        self.cols_digits = self.__count_digits(self.training)  # map <col,size>
        self.pos_columns = self.__column_checker('positive values', self.training, lambda x: x > 0)
        self.neg_columns = self.__column_checker('negative values', self.training, lambda x: x < 0)
        self.not_neg_columns = self.__column_checker('not negative values', self.training, lambda x: x >= 0)
        self.one_value_columns = self.__helper_K_possible_values_by_percentage('exactly one', self.training, coverage=1, percentage=1)  # map <col, value>
        self.cov_percen_columns = self.__helper_K_possible_values_by_percentage('coverage percentage columns',
                                                                                self.training, coverage=coverage,
                                                                                percentage=percentage)
        self.corr_columns = self.__corresponding_columns('corresponding columns', self.training)
        # print "ref"
        # for col in self.rules_fit.keys():
        #     if 'corresponding columns' in self.rules_fit[col]:
        #         self.count = self.count+1
        #     else:
        #         print self.rules_fit[col]
        # print self.count
        # print self.rules_fit

    def learnt_rules_anount(self):
        ans = ""
        ans += "digits rule: %s" % len(self.cols_digits)
        ans += "all-positive rule: %s" % len(self.pos_columns)
        ans += "all-negative rule: %s" % len(self.neg_columns)
        ans += "all non-negative rule: %s" % len(self.not_neg_columns)
        ans += "exactly one value rule: %s" % len(self.one_value_columns)
        ans += "coverage-percentage rule: %s" % len(self.cov_percen_columns)
        ans += "correlation rule: %s" % len(self.corr_columns)
        return ans

    def __corresponding_columns(self, method, dataset):
        def check_corresponding(col_i, col_j):
            ret = {}
            for _, row in dataset.iterrows():
                key = row[col_i]
                value = row[col_j]
                if key in ret:
                    if value != ret[key]:
                        return {}
                else:
                    if value in ret.values():
                        return {}
                    else:
                        ret[key] = value
            return ret

        one_value_columns = self.one_value_columns
        if one_value_columns == None:
            one_value_columns = self.__helper_K_possible_values_by_percentage('', self.training, coverage=1, percentage=1)
        columns = [col for col in dataset.columns if col not in one_value_columns]
        ret, skip = {}, []
        i = 0
        for col_i in columns:
            if not col_i in skip:
                if len(dataset[col_i].unique()) < 10:
                    for col_j in columns[i + 1:]:
                        if not col_j in skip:
                            mapping = check_corresponding(col_i, col_j)
                            if len(mapping) > 0:
                                # val = None
                                # for item in mapping.items():
                                #     if item[0] == 0:
                                #         continue
                                #     if val is None:
                                #         val = item[1] / item[0]
                                #     elif item[1] / item[0] != val:
                                #         continue
                                ret[(col_i, col_j)] = mapping
                                dic_col_j = {}
                                dic_col_j[col_j] = mapping
                                if col_i in self.rules_fit:
                                    if method in self.rules_fit[col_i]:
                                        self.rules_fit[col_i][method].update(dic_col_j)
                                    else:
                                        self.rules_fit[col_i][method] = {}
                                        self.rules_fit[col_i][method] = dic_col_j
                                else:
                                    self.rules_fit[col_i] = {}
                                    self.rules_fit[col_i][method] = dic_col_j
                                skip.append(col_j)
            i += 1
        return ret

    def get_removable_columns(self):
        from sets import Set
        removable_cols = Set(self.one_value_columns.keys())
        for (col_i,col_j) in self.corr_columns:
            removable_cols.add(col_i)
        return list(removable_cols)

    def transform_corresponding_columns(self, type, dataset):
        method_name = "corresponding columns"
        def corr(multi_columns, row, col_i, col_j):
            values = multi_columns[(col_i, col_j)]
            for val_i in values:
                val_j = values[val_i]
                if row[col_i] == val_i and row[col_j] != val_j:
                    return False
            return True
        prediction = []
        for _, row in dataset.iterrows():
            pred = Conf.POSITIVE_LABEL
            for (col_i,col_j) in self.corr_columns:
                if not corr(self.corr_columns, row, col_i, col_j):
                    pred = Conf.NEGATIVE_LABEL
                    # self.__update_rules_stat(col_i, method_name, col_j)
                    break
            prediction.append(pred)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    # our transform returns new_dataset and its values
    def transform_same_digits_number(self, type, dataset):
        '''
    checks foreach column if its values must contain V digits
    '''
        method_name = "same digits number"
        prediction = [[Conf.POSITIVE_LABEL] * len(dataset)]
        for topic_feature in self.cols_digits.keys():
            dig = self.cols_digits[topic_feature]
            prediction.append(self.__column_checker_on_testing(dataset, method_name, [topic_feature], lambda x: self.__digit(x) == dig))
        prediction = self.__intersection_labeling(*prediction)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    def transform_positive_values(self, type, dataset):
        method_name = "positive values"
        pos_prediction = self.__column_checker_on_testing(dataset, method_name, self.pos_columns, lambda x: x > 0)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(pos_prediction))
        return pos_prediction

    def transform_negative_values(self, type, dataset):
        method_name = "negative values"
        neg_prediction = self.__column_checker_on_testing(dataset, method_name, self.neg_columns, lambda x: x < 0)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(neg_prediction))
        return neg_prediction

    def transform_not_negative_values(self, type, dataset):
        method_name = "not negative values"
        no_neg_prediction = self.__column_checker_on_testing(dataset, method_name, self.not_neg_columns, lambda x: x >= 0)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(no_neg_prediction))
        return no_neg_prediction

    def transform_exactly_one_value(self, type, dataset):
        method_name = "exactly one"
        prediction = self.__checking_columns_values(dataset, method_name, self.one_value_columns)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    def transform_coverage_percentage_columns(self, type, dataset):
        method_name = "coverage percentage columns"
        prediction = self.__checking_columns_values(dataset, method_name, self.cov_percen_columns)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        # print prediction
        return prediction

    def __update_methods_stat(self, type, key, values):
        pos_value, neg_value = values
        try:
            self.methods_stats[type][key].update(pos_value, neg_value)
        except KeyError as e:
            self.methods_stats[type][key] = NegPos(pos_value, neg_value)

    # def __update_rules_stat(self, column_name, method, parameter_value=1):
    #     if isinstance(parameter_value, numpy.float64) and parameter_value > 9999999999999999:
    #         parameter_value = float('{:0.5e}'.format(parameter_value))
    #     if column_name in self.rules_stats:
    #         if method == 'coverage percentage columns' and method in self.rules_stats[column_name]:
    #             for item in parameter_value:
    #                 if item in self.rules_stats[column_name][method]:
    #                     pass
    #                 else:
    #                     self.rules_stats[column_name][method] = self.rules_stats[column_name][method] + item
    #         if method == "corresponding columns" and method not in self.rules_stats[column_name]:
    #             self.rules_stats[column_name] = {}
    #             self.rules_stats[column_name][method] = [parameter_value]
    #         elif method == "corresponding columns" and method in self.rules_stats[column_name]:
    #             self.rules_stats[column_name][method] = self.rules_stats[column_name][method] + [parameter_value]
    #             self.rules_stats[column_name][method] = list(set(self.rules_stats[column_name][method]))
    #             # self.rules_stats[column_name][method] = self.rules_stats[column_name][method] + parameter_value
    #         else:
    #             self.rules_stats[column_name][method] = parameter_value
    #     else:
    #         self.rules_stats[column_name] = {}
    #         self.rules_stats[column_name][method] = parameter_value

    def __update_deleted_topics_stat(self, topic, method):
        if topic in self.deleted_topics:
            self.deleted_topics[topic] += [method]
        else:
            self.deleted_topics[topic] = [method]

    def __count_negative_predictions(self, predication):
        neg_value = len(filter(lambda x: x == Conf.NEGATIVE_LABEL, predication))
        pos_value = len(predication) - neg_value
        return pos_value, neg_value

    def __intersection_labeling(self, labeling, *labelings):
        new_labeling = []
        labelings = list(labelings)
        labelings.append(labeling)
        for i in range(len(labeling)):
            lbl = True
            # count = 0
            for labeling in labelings:
                lbl = lbl and (labeling[i] == 1)
                # count += 1
                # print str(count) + ": " + str(labeling)
            new_labeling.append(Conf.POSITIVE_LABEL if lbl else Conf.NEGATIVE_LABEL)
        return new_labeling

    def __helper_K_possible_values_by_percentage(self, method, dataset, coverage, percentage):
        '''
        return a map of:
        each column which values it can get (only for columns with @uniques < counter)
        '''

        def my_unique(col_i, uniques):
            new_uniques = []
            tested_coverage = 0.0
            for uni in uniques:
                cnt = len(dataset[(dataset[col_i] == uni)]) + 0.0  # number of uni in the column
                rows = dataset.shape[ROWS_INDEX]
                if not (cnt / rows < percentage):
                    tested_coverage += (cnt / rows)
                    new_uniques.append(uni)
            return tested_coverage, new_uniques

        ret = {}
        for col_i in dataset.columns:
            uniques = dataset[col_i].unique()
            tested_coverage, uniques = my_unique(col_i, uniques)
            if len(uniques) > 0 and tested_coverage >= coverage:
                uni = []
                for val in uniques:
                    uni.append(self.__canonicalize_scientific_number(val))
                ret[col_i] = uniques
                if method == 'coverage percentage columns':
                    if col_i in self.rules_fit:
                        self.rules_fit[col_i][method] = uni
                    else:
                        self.rules_fit[col_i] = {}
                        self.rules_fit[col_i][method] = uni
                if method == 'exactly one':
                    if col_i in self.rules_fit:
                        self.rules_fit[col_i][method] = uni
                    else:
                        self.rules_fit[col_i] = {}
                        self.rules_fit[col_i][method] = uni
        return ret

    def __canonicalize_scientific_number(self, number):
        # for very big numbers
        if  number > 9999999999999999: #isinstance(number, numpy.float64) and
            number = float('{:0.5e}'.format(number))
        # for robil2
        # if number < 1:
        #     number = float('{:0.2e}'.format(number))
        # for arm_manipulation
        # if 0.7 < number < 0.71:
        #     number = float('{:0.10e}'.format(number))
        return number

    def __checking_columns_values(self, dataset, method, map_values):
        '''
    return dataset's labels according to map_values
    if the columns are the same values as @map_values.
    '''
        ret = []
        for _, row in dataset.iterrows():
            label = Conf.POSITIVE_LABEL
            for col_i in map_values.keys():
                if not (row[col_i] in map_values[col_i]):
                    label = Conf.NEGATIVE_LABEL
                    # print str(col_i) + " " + str(map_values[col_i])
                    self.__update_deleted_topics_stat(col_i, method)
                    break
                # else:
                #     print str(col_i) + " " + str(map_values[col_i])
            ret.append(label)
        # if method == 'coverage percentage columns' and not 0 in ret:
        #     values = map_values[col_i]  # [map_values[k][0] for k in map_values]
        #     values = list(set(values))
        #     self.__update_rules_stat(col_i, method, values)
        # elif method == 'exactly one' and not 0 in ret:
        #     self.__update_rules_stat(col_i, method, map_values[col_i])
        # elif not 0 in ret:
        #     self.__update_rules_stat(col_i, method, map_values)
        return ret

    def __column_checker(self, method, dataset, predicate):
        '''
    return columns that setisfy the given predicate
    '''
        columns = []
        for col_i in dataset:
            pred = True
            for _, row in dataset.iterrows():
                pred = pred and predicate(row[col_i])
                if not pred:
                    break
            if pred:
                columns.append(col_i)
                if col_i in self.rules_fit:
                    self.rules_fit[col_i][method] = 1
                else:
                    self.rules_fit[col_i] = {}
                    self.rules_fit[col_i][method] = 1
        return columns


    def __column_checker_on_testing(self, dataset, method, columns, predicate):
        '''
    checks that the following @columns setisfy the predicate
    '''
        # topic = ""
        labeling = []
        for _, row in dataset.iterrows():
            pred = Conf.POSITIVE_LABEL
            for col_i in columns:
                # topic = col_i
                # if "Messages-Age(/move_base/local_costmap/footprint)" == topic:
                #     print "yes"
                # topic = col_i
                # if "matansar(MaxCPU5)" == topic:
                #     print "yes"
                if not predicate(row[col_i]):
                    pred = Conf.NEGATIVE_LABEL
                    # self.__update_deleted_topics_stat(col_i, method)
                # else:
                #     if method == 'same digits number':
                #         pass
                #         self.__update_rules_stat(col_i, method, self.__digit(row[col_i]))
                #     else:
                #         self.__update_rules_stat(col_i, method)
            labeling.append(pred)
        # print str(topic) + ": " + str(labeling)
        # if method == 'same digits number' and not 0 in labeling:
        #     self.__update_rules_stat(col_i, method, self.__digit(row[col_i]))
        # elif not 0 in labeling:
        #     self.__update_rules_stat(col_i, method)
        return labeling

    def __count_digits(self, dataset):
        def count_digit_of(col_i):
            # print dataset[col_i]
            temp = -1
            for item in dataset[col_i]:
                # print self.__digit(item)
                if temp == -1:
                    temp = self.__digit(item)
                elif temp != self.__digit(item):
                    return -1
            return temp
            # min_digit = self.__digit(dataset[col_i].min())
            # max_digit = self.__digit(dataset[col_i].max())
            # if min_digit == max_digit:
            #     return min_digit
            # return -1
        ret = {}
        for col_i in dataset:
            # if "Messages-Age(/move_base/local_costmap/footprint)" == col_i:
            #     print "yes"
            dig = count_digit_of(col_i)
            if dig > 0:
                ret[col_i] = dig
                self.rules_fit[col_i] = {}
                self.rules_fit[col_i]['same digits number'] = dig
        return ret

    def __digit(self, number):
        # if isinstance(number, str):
        #     number = 9223372036854775807
        number = int(abs(number))
        return len(str(number))

    def __str__(self):
        def map_values(type, m):
            ret = "%s: " %(type)
            for num, k in zip(range(len(m)),m):
                ret += "%s)%s: %s " % (num, k, m[k])
            return ret + "\n"
        s = Style.BOLD + ("Role-Based Anomaly Detection: percentage=%s, coverage=%s\n" % (self.percentage, self.coverage)) + Style.END
        for type in self.methods_stats:
            s += map_values(type, self.methods_stats[type])
        s += "\thow much constant columns: %s/%s\n" % (len(self.one_value_columns), self.training.shape[COLOMUNS_INDEX])
        s += "\thow much correlate columns: %s/%s\n" % (len(self.corr_columns), self.training.shape[COLOMUNS_INDEX])
        print self.methods_stats["negatives"]
        return s