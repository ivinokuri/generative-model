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
        self.percentage = None
        self.coverage = None

    def fit(self, rules):

        self.bound = self.get_value_by_topic(rules, "bound")

        self.cols_digits = self.get_value_by_topic(rules, "same digits number")

        self.pos_columns = self.get_list_by_topic(rules, "positive values")

        self.neg_columns = self.get_list_by_topic(rules, "negative values")
        self.not_neg_columns = self.get_list_by_topic(rules, "not negative values")


        self.one_value_columns = self.get_value_by_topic(rules, "exactly one")
        self.cov_percen_columns = self.get_value_by_topic(rules, "coverage percentage columns")
        #
        self.corr_columns = self.get_corr_topics(rules, "corresponding columns")
        # self.min_max_values = self.__calculate_min_max_columns_values(self.training)  # map<col,<min,max> >
        # print self.pos_columns

    def get_value_by_topic(self, rules, key):
        ret = {}
        for val in rules.keys():
            for key_val in rules[val].keys():
                if key_val == key:
                    result = {}
                    result = rules[val][key_val]
                    ret[val.encode("utf-8")] = result
        return ret

    def get_list_by_topic(self, rules, key):
        topic_list = []
        for val in rules.keys():
            for key_val in rules[val].keys():
                if key_val == key:
                    topic_list.append(val.encode("utf-8"))
        return topic_list

    def get_corr_topics(self, rules, key):
        ret = {}
        for val in rules.keys():
            for key_val in rules[val].keys():
                if key_val == key:
                    for key_val_inside in rules[val][key_val].keys():
                        re = {}
                        for key_val_inside_result in rules[val][key_val][key_val_inside].keys():
                            re[key_val_inside_result.encode("utf-8")] = rules[val][key_val][key_val_inside][key_val_inside_result]
                        ret[(val.encode("utf-8"), key_val_inside.encode("utf-8"))] = re
        return ret

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

    def __corresponding_columns(self, dataset):
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
            one_value_columns = self.__helper_K_possible_values_by_percentage(self.training, coverage=1, percentage=1)
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
                                # print str(col_i) + " " + str(col_j) + str(ret[(col_i, col_j)])
                                skip.append(col_j)
            i += 1
        return ret

    def get_removable_columns(self):
        from sets import Set
        removable_cols = Set(self.one_value_columns.keys())
        for (col_i,col_j) in self.corr_columns:
            removable_cols.add(col_i)
        return list(removable_cols)

    # def transform_corresponding_columns(self, type, dataset, rules):
    #     method_name = "corresponding columns"
    #     labeling = []
    #     def corr(multi_columns, row, col_i, col_j):
    #         values = multi_columns[(col_i, col_j)]
    #         for val_i in values:
    #             val_j = values[val_i]
    #             if row[col_i] == val_i and row[col_j] != val_j:
    #                 return False
    #         return True
    #     prediction = []
    #     for _, row in dataset.iterrows():
    #         pred = Conf.POSITIVE_LABEL
    #         for (col_i,col_j) in self.corr_columns:
    #             if not corr(self.corr_columns, row, col_i, col_j):
    #                 pred = Conf.NEGATIVE_LABEL
    #                 if not col_i in rules[col_i][method_name]:
    #                     labeling.append(Conf.NEGATIVE_LABEL)
    #                 break
    #             else:
    #                 labeling.append(Conf.POSITIVE_LABEL)
    #         prediction.append(pred)
    #     self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
    #     print "Corresponding results : " + labeling
    #     return prediction

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
                    # if type == "negatives":
                    # print method_name + "\t" + str(col_i) + "\t" + str(row[col_i])
                    break
            prediction.append(pred)
        # self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction


    def is_rule_considered(self, rules, feature, method_name):
        return feature in rules and method_name in rules[feature]

    # our transform returns new_dataset and its values
    # def transform_same_digits_number(self, type, dataset, rules):
    #     '''
    # checks foreach column if its values must contain V digits
    # '''
    #     method_name = "same digits number"
    #     prediction = [[Conf.POSITIVE_LABEL] * len(dataset)]
    #     for col_i in self.cols_digits.keys():
    #         if not self.is_rule_considered(rules, col_i, method_name):
    #             continue
    #         dig = self.cols_digits[col_i]
    #         rules_learned_digit = rules[col_i][method_name]
    #         prediction.append(self.__column_checker_on_testing(dataset, [col_i], lambda x: self.__digit(x) == dig and self.__digit(x) == rules_learned_digit))
    #     prediction = self.__intersection_labeling(*prediction)
    #     self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
    #     print prediction
    #     return prediction

    # our transform returns new_dataset and its values
    def transform_bound(self, type, dataset):
        '''
    checks foreach column if its values must contain V digits
    '''
        prediction = [[Conf.POSITIVE_LABEL] * len(dataset)]
        for col_i in self.bound.keys():
            var = self.bound[col_i]
            for item in var.keys():
                if item == "high":
                    high = var[item]
                elif item == "low":
                    low = var[item]
                elif item == "abs":
                    abs = var[item]
            prediction.append(self.__column_checker_on_bound(dataset, [col_i], high, low, abs))
        prediction = self.__intersection_labeling(*prediction)
        return prediction

    def transform_same_digits_number(self, type, dataset):
        '''
    checks foreach column if its values must contain V digits
    '''
        prediction = [[Conf.POSITIVE_LABEL] * len(dataset)]
        for col_i in self.cols_digits.keys():
            dig = self.cols_digits[col_i]
            prediction.append(self.__column_checker_on_testing(dataset, [col_i], lambda x: self.__digit(x) == dig))
        # prediction = self.__intersection_labeling(*prediction)
        return prediction

    def transform_positive_values(self, type, dataset):
        pos_prediction = self.__column_checker_on_testing(dataset, self.pos_columns, lambda x: x > 0)
        return pos_prediction

    def transform_negative_values(self, type, dataset):
        neg_prediction = self.__column_checker_on_testing(dataset, self.neg_columns, lambda x: x < 0)
        print neg_prediction
        return neg_prediction

    def transform_not_negative_values(self, type, dataset):
        no_neg_prediction = self.__column_checker_on_testing(dataset, self.not_neg_columns, lambda x: x >= 0)
        print no_neg_prediction
        return no_neg_prediction

    def transform_exactly_one_value(self, type, dataset):
        prediction = self.__checking_columns_values(dataset, self.one_value_columns)
        print prediction
        return prediction

    def transform_coverage_percentage_columns(self, type, dataset):
        prediction = self.__checking_columns_values(dataset, self.cov_percen_columns)
        print prediction
        return prediction

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

    def __helper_K_possible_values_by_percentage(self, dataset, coverage, percentage):
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
                ret[col_i] = uniques
        return ret

    def __canonicalize_scientific_number(self, number):
        if number > 9999999999999999: #isinstance(number, numpy.float64) and
            number = float('{:0.5e}'.format(number))
        if number < 0.02:
            number = float('{:0.2e}'.format(number))
        if 0.7 < number < 0.71:
            number = float('{:0.10e}'.format(number))
        return number

    def  __checking_columns_values(self, dataset, map_values):
        '''
    return dataset's labels according to map_values
    if the columns are the same values as @map_values.
    '''
        ret = []
        for _, row in dataset.iterrows():
            label = Conf.POSITIVE_LABEL
            for col_i in map_values.keys():
                row[col_i] = self.__canonicalize_scientific_number(row[col_i])
                if not (row[col_i] in map_values[col_i]):
                    # if not "/rosout" in col_i:
                    label = Conf.NEGATIVE_LABEL
                    # if type == "negatives":
                    # print method_name + "\t" + str(col_i) + "\t" + str(row[col_i]) + "\t not in " + str(map_values[col_i])
                    break
            ret.append(label)
        return ret

    # "/rosout" and "/twist_mux"

    # def __checking_columns_values(self, dataset, map_values, method, rules):
    #     '''
    #         return dataset's labels according to map_values
    #         if the columns are the same values as @map_values.
    #     '''
    #     ret = []
    #     for _, row in dataset.iterrows():
    #         label = Conf.POSITIVE_LABEL
    #         for col_i in map_values.keys():
    #             if isinstance(row[col_i], numpy.float64) and row[col_i] > 9999999999999999:
    #                 row[col_i] = self.__canonicalize_scientific_number(row[col_i])
    #             # if row[col_i] == 9.2233720368499999e+18:
    #             #     continue
    #             if not self.is_rule_considered(rules, col_i, method):
    #                 continue
    #             canonical_column = [self.__canonicalize_scientific_number(val) for val in rules[col_i][method]]
    #             # for index in xrange(len(rules[col_i][method])):
    #             #     rules[col_i][method][index] = self.__canonicalize_scientific_number(rules[col_i][method][index])
    #             if not (row[col_i] in canonical_column):
    #                 label = Conf.NEGATIVE_LABEL
    #                 break
    #
    #         ret.append(label)
    #     return ret

    def __column_checker(self, dataset, predicate):
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
        return columns


    # def __column_checker_on_testing(self, dataset, columns, predicate, method_name="", rules=""):
    #     '''
    # checks that the following @columns setisfy the predicate
    # '''
    #     # topic = ""
    #     labeling = []
    #     for _, row in dataset.iterrows():
    #         pred = Conf.POSITIVE_LABEL
    #         last_col = None
    #         for col_i in columns:
    #             if rules != "" and not self.is_rule_considered(rules, col_i, method_name):
    #                 break
    #             if not predicate(row[col_i]):
    #                 pred = Conf.NEGATIVE_LABEL
    #             elif rules != "":
    #                 pred = pred * rules[col_i][method_name]
    #
    #         labeling.append(pred)
    #     # print str(topic) + ": " + str(labeling)
    #     return labeling

    def __column_checker_on_testing(self, dataset, columns, predicate):
        '''
    checks that the following @columns setisfy the predicate
    '''
        labeling = []
        for _, row in dataset.iterrows():
            pred = Conf.POSITIVE_LABEL
            for col_i in columns:
                if not predicate(row[col_i]):
                    print str(col_i) + " " + str(row[col_i])
                    pred = Conf.NEGATIVE_LABEL
            labeling.append(pred)
        return labeling

    def __column_checker_on_bound(self, dataset, columns, high, low, abs):
        '''
    checks that the following @columns setisfy the predicate
    '''
        labeling = []
        for _, row in dataset.iterrows():
            pred = Conf.POSITIVE_LABEL
            for col_i in columns:
                if not self.__check_by_bound(row[col_i], high, low, abs):
                    print str(col_i) + " " + str(row[col_i])
                    pred = Conf.NEGATIVE_LABEL
            labeling.append(pred)
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
        return ret

    def __calculate_min_max_columns_values(self, dataset):
        min_max_values = {}
        for col_i in dataset.columns:
            a, b = self.__get_min_max_values(dataset, col_i)
            if self.__fulfill_column_constains(a, b):
                min_max_values[col_i] = (a, b)
        return min_max_values

    def __check_bit_column_values(self, dataset, col_i):
        # if "Messages-Age(/move_base/status)" == col_i:
        #     print "yes"
        labeling = []
        a, b = self.min_max_values[col_i]
        for _, row in dataset.iterrows():
            value = row[col_i]
            if not self.__fulfill_value_constains(a, b, value):
                labeling.append(Conf.NEGATIVE_LABEL)
            else:
                labeling.append(Conf.POSITIVE_LABEL)
        # print str(col_i) + ": " + str(labeling)
        return labeling

    def __get_min_max_values(self, dataset, col_i):
        a = dataset[col_i].min()
        b = dataset[col_i].max()
        return a, b

    def __fulfill_column_constains(self, a, b, distance=1):
        digit_a, digit_b = self.__digit(a), self.__digit(b)
        if digit_b - digit_a <= distance:
            return True
        return False

    def __fulfill_value_constains(self, a, b, value):
        digit_a, digit_b = self.__digit(a), self.__digit(b)
        digit_val = self.__digit(value)
        if digit_a > digit_val or digit_val > digit_b:
            return False
        distance = pow(10, self.__digit(b - a) - 1)
        # if (self.__digit(b - a) > 7 ):
        #     print "diff ", str(b-a)
        #     print "digit ", str(self.__digit(b - a) - 1)
        #     print "distance ", str(distance)
        if a - distance > value or value - distance > b: # overflow long warning
            return False
        return True

    def __digit(self, number):
        # if isinstance(number, str):
        #     number = 9223372036854775807
        number = int(abs(number))
        return len(str(number))

    def __check_by_bound(self, number, high, low, abs1):
        result = False
        if abs1 == 1:
            if low <= abs(number) < high:
                result = True
        else:
            if low <= number < high:
                result = True
        return result

    def __str__(self):
        def map_values(type, m):
            ret = "\t%s:\n" %(type)
            for num, k in zip(range(len(m)),m):
                ret += "\t\t%s) method: %s\n\t\t%sscore: %s\n" % (num, k, ' ' * (len(str(num)) + 2), m[k])
            return ret
        s = Style.BOLD + ("Role-Based Anomaly Detection: percentage=%s, coverage=%s\n" % (self.percentage, self.coverage)) + Style.END
        for type in self.methods_stats:
            s += map_values(type, self.methods_stats[type])
        s += "\thow much constant columns: %s/%s\n" % (len(self.one_value_columns), self.training.shape[COLOMUNS_INDEX])
        s += "\thow much correlate columns: %s/%s\n" % (len(self.corr_columns), self.training.shape[COLOMUNS_INDEX])
        print self.methods_stats["negatives"]
        return s