import pandas as pd
from Style import Style, Configure
from Style import Configure as Conf

ROWS_INDEX = 0
COLOMUNS_INDEX = 1


class NegPos:

    def __init__(self, pos, neg):
        self.pos = pos
        self.neg = neg

    def update(self, pos, neg):
        self.pos += pos
        self.neg += neg

    def __str__(self):
        return "positives = %s, negatives = %s" % (self.pos, self.neg)


class SequenceClass:

    def __init__(self, col_i, method_name, line_number):
        self.title = []
        self.title.append(method_name + "-" + col_i + ": " + str(line_number))

    def update1(self, col_i, method_name, line_number):
        self.title.append(method_name + "-" + col_i + ": " + str(line_number))


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
        self.dictionary_longest = None

    def fit(self, trainings, percentage, coverage):

        self.dictionary_longest = {}

        self.percentage = percentage
        self.coverage = coverage
        self.training = pd.concat(trainings, ignore_index=True)

        self.cols_digits = self.__count_digits(self.training)  # map <col,size>
        self.pos_columns = self.__column_checker(self.training, lambda x: x > 0)
        self.neg_columns = self.__column_checker(self.training, lambda x: x < 0)
        self.not_neg_columns = self.__column_checker(self.training, lambda x: x >= 0)
        self.one_value_columns = self.__helper_K_possible_values_by_percentage(self.training, coverage=1, percentage=1)  # map <col, value>
        self.cov_percen_columns = self.__helper_K_possible_values_by_percentage(self.training, coverage=coverage, percentage=percentage)

        self.corr_columns = self.__corresponding_columns(self.training)
        self.min_max_values = self.__calculate_min_max_columns_values(self.training)  # map<col,<min,max> >

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
                                ret[(col_i, col_j)] = mapping
                                skip.append(col_j)
            i += 1
        return ret

    def get_removable_columns(self):
        from sets import Set
        removable_cols = Set(self.one_value_columns.keys())
        for (col_i,col_j) in self.corr_columns:
            removable_cols.add(col_i)
        return list(removable_cols)

    def transform_corresponding_columns(self, name, type, dataset):
        method_name = "corresponding columns"
        def corr(multi_columns, row, col_i, col_j):
            values = multi_columns[(col_i, col_j)]
            for val_i in values:
                val_j = values[val_i]
                if row[col_i] == val_i and row[col_j] != val_j:
                    return False
            return True
        prediction = []
        i = 0
        for _, row in dataset.iterrows():
            x = dataset.index.values[i]
            pred = Conf.POSITIVE_LABEL
            for (col_i,col_j) in self.corr_columns:
                if not corr(self.corr_columns, row, col_i, col_j):
                    pred = Conf.NEGATIVE_LABEL
                    # print "csv type: " + type + " - " + name + ": " + method_name + ": " + col_i + " and " + col_j + " not matches - values: " + str(self.corr_columns[(col_i, col_j)]) + " in line: " + str(x)
                    self.__print_sequence(name, type, method_name, col_i, x)
                    break
            i = i + 1
            prediction.append(pred)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    # our transform returns new_dataset and its values
    def transform_same_digits_number(self,name ,type, dataset):
        '''
    checks foreach column if its values must contain V digits
    '''
        method_name = "same digits number"
        prediction = [[Conf.POSITIVE_LABEL] * len(dataset)]
        for col_i in self.cols_digits.keys():
            dig = self.cols_digits[col_i]
            prediction.append(self.__column_checker_on_testing(dig, type, name, method_name, dataset, [col_i], lambda x: self.__digit(x) == dig))
        prediction = self.__intersection_labeling(*prediction)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    def transform_positive_values(self,name ,type , dataset):
        method_name = "positive values"
        pos_prediction = self.__column_checker_on_testing("", type, name, method_name, dataset, self.pos_columns, lambda x: x > 0)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(pos_prediction))
        return pos_prediction

    def transform_negative_values(self,name ,type , dataset):
        method_name = "negative values"
        neg_prediction = self.__column_checker_on_testing("", type, name, method_name, dataset, self.neg_columns, lambda x: x < 0)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(neg_prediction))
        return neg_prediction

    def transform_not_negative_values(self, name, type, dataset):
        method_name = "not negative values"
        no_neg_prediction = self.__column_checker_on_testing("", type, name, method_name, dataset, self.not_neg_columns, lambda x: x >= 0)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(no_neg_prediction))
        return no_neg_prediction

    def transform_exactly_one_value(self, name, type, dataset):
        method_name = "exactly one"
        prediction = self.__checking_columns_values(type, name, method_name, dataset, self.one_value_columns)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    def transform_coverage_percentage_columns(self, name, type, dataset):
        method_name = "coverage percentage columns"
        prediction = self.__checking_columns_values(type, name, method_name, dataset, self.cov_percen_columns)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    def transform_bit_same_column_values(self, type, dataset):
        method_name = "a bit same values"
        prediction = [Conf.POSITIVE_LABEL] * dataset.shape[ROWS_INDEX]  # all samples are ok, until we find something
        for col_i in self.min_max_values.keys():
            a, b = self.min_max_values[col_i]
            if self.__fulfill_column_constains(a, b):
                new_labeling = self.__check_bit_column_values(dataset, col_i)
                prediction = self.__intersection_labeling(prediction, new_labeling)
        self.__update_methods_stat(type, method_name, self.__count_negative_predictions(prediction))
        return prediction

    def __update_methods_stat(self, type, key, values):
        pos_value, neg_value = values
        try:
            self.methods_stats[type][key].update(pos_value, neg_value)
        except KeyError as e:
            self.methods_stats[type][key] = NegPos(pos_value, neg_value)

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
            for labeling in labelings:
                lbl = lbl and (labeling[i] == 1)
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

    def __checking_columns_values(self, type, name, method_name, dataset, map_values):
        '''
    return dataset's labels according to map_values
    if the columns are the same values as @map_values.
    '''
        ret = []
        i = 0
        for _, row in dataset.iterrows():
            x = dataset.index.values[i]
            label = Conf.POSITIVE_LABEL
            for col_i in map_values.keys():
                if not (row[col_i] in map_values[col_i]):
                    label = Conf.NEGATIVE_LABEL
                    # print "csv type: " + type + " - " + name + ": " + method_name + " in: " + col_i + " value: " + str(
                    #     row[col_i]) + " not in: " + str(map_values[col_i]) + " in line: " + str(x)
                    self.__print_sequence(name, type, method_name, col_i, x)
                    break
            i = i + 1
            ret.append(label)
        return ret

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

    def __column_checker_on_testing(self, dig, type, name, method_name, dataset, columns, predicate):
        '''
    checks that the following @columns setisfy the predicate
    '''
        labeling = []
        i = 0
        for _, row in dataset.iterrows():
            x = dataset.index.values[i]
            pred = Conf.POSITIVE_LABEL
            for col_i in columns:
                if not predicate(row[col_i]):
                    pred = Conf.NEGATIVE_LABEL
                    result = "csv type: " + type + " - " + name + ": " + method_name + " in: " + col_i + " value: " + str(row[col_i]) + " in line: " + str(x)
                    if dig != "":
                        result = result + " real same digit: " + str(dig)
                    #print result
                    self.__print_sequence(name, type, method_name, col_i, x)
            i = i + 1
            labeling.append(pred)
        return labeling

    def __print_sequence(self, name, type, method_name, col_i, line_number):
        self.__update__stat(name, type, method_name, col_i, line_number)

    def __update__stat(self, name, type, method_name, col_i, line_number):
        try:
            self.dictionary_longest[type+"-"+name].update1(col_i, method_name, line_number)
        except KeyError as e:
            self.dictionary_longest[type+"-"+name] = SequenceClass(col_i, method_name, line_number)

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
        labeling = []
        a, b = self.min_max_values[col_i]
        for _, row in dataset.iterrows():
            value = row[col_i]
            if not self.__fulfill_value_constains(a, b, value):
                labeling.append(Conf.NEGATIVE_LABEL)
            else:
                labeling.append(Conf.POSITIVE_LABEL)
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
        for name, item in sorted(self.dictionary_longest.items(), reverse=True):
            print name + ": " + str(item.title)
        return s