import RuleBasedAnomaly as RBA
import Dataset as DS
import DatasetNew as NEWDS
import Measurements as M
from Style import Configure as Conf
import numpy as np

global options


def RBA_stage(datasets_preds):
    def apply_rba(rba, type, datasets):
        datasets_predictions = []
        removable_cols = rba.get_removable_columns()
        new_datasets = []
        for dataset in datasets:

            # pred = rba.transform_same_digits_number(type, dataset)
            # pred_original = pred
            #
            # print "same digits number: " + str(__count_negative_predictions(pred_original))
            #
            # pred_original1 = rba.transform_positive_values(type, dataset)
            # pred = datasets_preds.intersection_labeling(pred, pred_original1)
            #
            # print "positive values: " + str(__count_negative_predictions(pred_original1))
            #
            # pred_original2 = rba.transform_negative_values(type, dataset)
            # pred = datasets_preds.intersection_labeling(pred, pred_original2)
            #
            # print "negative values: " + str(__count_negative_predictions(pred_original2))
            #
            # pred_original3 = rba.transform_not_negative_values(type, dataset)
            # pred = datasets_preds.intersection_labeling(pred, pred_original3)
            #
            # print "not negative values: " + str(__count_negative_predictions(pred_original3))
            #
            # pred_original4 = rba.transform_coverage_percentage_columns(type, dataset)
            # pred = datasets_preds.intersection_labeling(pred, pred_original4)
            #
            # print "coverage percentage columns: " + str(__count_negative_predictions(pred_original4))
            #
            # pred_original5 = rba.transform_corresponding_columns(type, dataset)
            # pred = datasets_preds.intersection_labeling(pred, pred_original5)
            #
            # print "corresponding columns: " + str(__count_negative_predictions(pred_original5))
            #
            # pred_original6 = rba.transform_exactly_one_value(type, dataset)
            # pred = datasets_preds.intersection_labeling(pred, pred_original6)
            #
            # print "exactly one: " + str(__count_negative_predictions(pred_original6))

            pred = rba.transform_same_digits_number(type, dataset)
            pred = datasets_preds.intersection_labeling(pred, rba.transform_positive_values(type, dataset))
            pred = datasets_preds.intersection_labeling(pred, rba.transform_negative_values(type, dataset))
            pred = datasets_preds.intersection_labeling(pred, rba.transform_not_negative_values(type, dataset))
            pred = datasets_preds.intersection_labeling(pred, rba.transform_coverage_percentage_columns(type, dataset))
            pred = datasets_preds.intersection_labeling(pred, rba.transform_corresponding_columns(type, dataset))
            pred = datasets_preds.intersection_labeling(pred, rba.transform_exactly_one_value(type, dataset))
            # pred = datasets_preds.intersection_labeling(pred, rba.transform_bit_same_column_values(type, dataset))
            datasets_predictions.append(pred)
            new_datasets.append(DS.Datasets.remove_columns(dataset, removable_cols))
        return new_datasets, datasets_predictions

    trainings, positives, negatives = datasets_preds.get_datasets()
    rba = RBA.RuleBasedAnomaly()

    rba.fit(trainings, percentage=0.01, coverage=0.98)
    training_datasets, trainings_preds = apply_rba(rba, Conf.TRAINING, trainings)
    positive_datasets, positives_preds = apply_rba(rba, Conf.POSITIVE, positives)
    negative_datasets, negatives_preds = apply_rba(rba, Conf.NEGATIVE, negatives)

    datasets_preds.update_predictions(trainings_preds, positives_preds, negatives_preds)
    datasets_preds.set_datasets(training_datasets, positive_datasets, negative_datasets)
    return datasets_preds, str(rba)


# def __count_negative_predictions(predication):
#     neg_value = len(filter(lambda x: x == Conf.NEGATIVE_LABEL, predication))
#     pos_value = len(predication) - neg_value
#     return pos_value, neg_value


def start(positives_dir_path, experiments, test_size, average_limit, negatives_dir_path, new_dir_path):
    global array_average_percent
    global min_acc_array
    global min_acc_array_training
    global min_acc_array_positive
    time_dir_paths = DS.Datasets.get_dirs_names(positives_dir_path)
    # for simulator_gazebo
    delete = 4
    # for robil2_gazebo
    # delete = 1
    time_dir_paths.sort(reverse=True)
    print time_dir_paths
    time_percent = {}
    for time_dir in time_dir_paths:
        time_str = time_dir.rsplit('/', 1)[0].rsplit('/', 1)[1]
        for i in range(experiments):
            # for skip checking of best window time
            # array_average_percent.append(0.90)
            check_percent(delete, time_str, time_dir, test_size)
            min_acc_array_training = []
            min_acc_array_positive = []
        average = np.mean(array_average_percent)
        if min_acc_array:
            min_accu = min(min_acc_array)
        else:
            min_accu = 1.1
        print "average: " + str(average)
        time_percent[time_dir] = average
        if average >= average_limit:
            print "accuracy array: " + str(min_acc_array)
            print "minimum accuracy: " + str(min_accu)
            next_step(delete, time_str, time_dir, negatives_dir_path+str(time_str), new_dir_path, min_accu)
            break
        array_average_percent = []
        min_acc_array = []
    time_percent = sorted(time_percent.items(), key=lambda x: x[1])
    print "time & average: " + str(time_percent)


def check_percent(delete, time_str, time_dir, test_size):
    datasets_preds = DS.Datasets(time_dir, [], test_size=test_size, delete=int(delete / float(time_str)))
    datasets_preds, rba_info = RBA_stage(datasets_preds)
    print results(datasets_preds, 0, rba_info=rba_info)


def next_step(delete, time_str, time_dir, negatives_dir_path, new_dir_path, min_accu):
    datasets_preds = NEWDS.DatasetsNew(time_dir, negatives_dir_path+"/", new_dir_path, time_str, delete=int(delete / float(time_str)))
    datasets_preds, rba_info = RBA_stage(datasets_preds)
    print results(datasets_preds, min_accu, rba_info=rba_info)


def predictions_information(datasets_preds, min_accu):
    def get_acc_info(title, paths_preds, pred_value, max_long, min_accu):
        ans = "-----%s-----:" % (title,)
        tmp = ""
        longests = 0.0
        anomalies = 0.0
        warnings = 0.0
        faulty_runs = 0.0
        acc_size = 5
        f_size = 30
        # min_acc = 1.0
        min_acc_pos = 1.0
        min_acc_training = 1.0
        for f, y_pred in paths_preds:
            diff = len(y_pred)/4
            y_true = [pred_value] * len(y_pred)
            acc1 = round(M.Measurements.accuracy_score(y_true, y_pred), acc_size)
            # if acc1 < min_acc:
            #     min_acc = acc1
            if title == 'training sets':
                if acc1 < min_acc_training:
                    min_acc_training = acc1
            if title == 'positive sets':
                if acc1 < min_acc_pos:
                    min_acc_pos = acc1
            acc = str(acc1) + " (" + \
                  str(len(filter(lambda x: x == pred_value, y_pred[:diff]))) + "/" + str(len(y_pred[:diff])) + ") (" + \
                  str(len(filter(lambda x: x == pred_value, y_pred[:diff*2]))) + "/" + str(len(y_pred[:diff*2])) + ") (" + \
                  str(len(filter(lambda x: x == pred_value, y_pred[:diff*3]))) + "/" + str(len(y_pred[:diff*3])) + ") (" + \
                  str(len(filter(lambda x: x == pred_value, y_pred))) + "/" + str(len(y_pred)) + ")"
            counter = 0
            number_of_zero = 0
            max_number_of_zeros = 0
            for line in y_pred:
                number_of_zero += 1 - line
                counter += 1
                if counter == 10:
                    if number_of_zero > max_number_of_zeros:
                        max_number_of_zeros = number_of_zero
                    number_of_zero = 0
                    counter = 0
            acc = acc + " most negative: " + str(max_number_of_zeros)
            if title == 'positive sets' or title == 'negative sets':
                if acc1 < min_accu and min_accu != 1.1:
                    acc = acc + " (below min accuracy)"
            anomalies += len(filter(lambda x: x != pred_value, y_pred))
            longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
            if longest > max_long:
                acc = acc + " (threshold: " + str(longest-max_long) + " more)"
            acc = acc.ljust(100)
            longest_str = str(longest).ljust(6)
            longests += longest
            warning = M.Measurements.count_warnings(y_pred, Conf.NEGATIVE_LABEL, int(max_long + 1))
            if warning > 0:
                faulty_runs += 1
            warnings += warning
            tmp += "%s%s\taccuracy: %slongest negative: %swarnings = %s\n" %(f, " " * (f_size - (len(f)-2)), acc, longest_str, warning)
            if negatives_names:
                M.Measurements.draw_chart(f, options.charts + f[:-4], y_pred, Conf.NEGATIVE_LABEL)
        ans = ans + "longest average = %s, anomalies = %s,  total warnings = %s,  faulty runs = %s,  non-faulty runs = %s, max threshold = %s\n" % (longests/len(paths_preds), anomalies, warnings, faulty_runs, len(paths_preds) - faulty_runs, str(max_long))
        non_faulty = len(paths_preds) - faulty_runs
        percent = non_faulty/len(paths_preds)
        if title == 'positive sets':
            array_average_percent.append(percent)
            min_acc_array.append(min_acc_pos)
            min_acc_array_positive.append(min_acc_pos)
        if title == 'training sets':
            min_acc_array_training.append(min_acc_training)
        if min_acc_array_training:
            min_training = min(min_acc_array_training)
            if title == 'positive sets':
                ans = ans + "accuarcy min trainings: " + str(min_training)
        if min_acc_array_positive:
            min_positve = min(min_acc_array_positive)
            if title == 'positive sets':
                ans = ans + ", accuarcy min positives: " + str(min_positve)
        if min_acc_array_training and min_acc_array_positive:
            if title == 'positive sets':
                if min_training == 0:
                    ans = ans + ", ratio: 0\n"
                else:
                    ans = ans + ", ratio: " + str(min_positve/min_training) + "\n"

        return ans + tmp
    trainings_preds, positives_preds, negatives_preds = datasets_preds.get_predictions()
    trainings_names, positives_names, negatives_names = datasets_preds.get_names()
    res = ""
    max_long = 0.0
    for f1, y_pred1 in zip(trainings_names, trainings_preds):
        longest1 = M.Measurements.longest_sequence(y_pred1, Conf.NEGATIVE_LABEL)
        if longest1 > max_long:
            max_long = longest1
    # print "max threshold " + str(max_long)
    res += get_acc_info("training sets", zip(trainings_names, trainings_preds), Conf.POSITIVE_LABEL, max_long, min_accu) + "\n"
    res += get_acc_info("positive sets", zip(positives_names, positives_preds), Conf.POSITIVE_LABEL, max_long, min_accu) + "\n"
    if negatives_names:
        res += get_acc_info("negative sets", zip(negatives_names, negatives_preds), Conf.POSITIVE_LABEL, max_long, min_accu) + "\n"
    return res


def results(datasets_preds, min_accu, rba_info):
    info = predictions_information(datasets_preds, min_accu)
    s = str(datasets_preds) + "\n"
    s += rba_info + "\n"
    s += info + "\n"
    return s


def ApplyOptions(run):
    from optparse import OptionParser
    defult_positives = ""#get_files_name("/home/matansar/thesis/available runs/manipulation scenarios/%s/normal/" % (run,))
    defult_charts = "/home/matansar/thesis/available runs/manipulation scenarios/%s/charts/" % (run,)
    parser = OptionParser()
    parser.add_option("-p", "--positives", dest="positives", help="positives directory path", default=defult_positives, metavar="POSITIVES")
    parser.add_option("-n", "--negatives", dest="negatives", help="negatives directory path", metavar="NEGATIVES")
    parser.add_option("-e", "--experiments", dest="experiments", help="experiments", default=4, metavar="EXPERIMENTS", type="int")
    parser.add_option("-t", "--test_size", dest="test_size", help="test_size", default=0.25, metavar="TEST_SIZE", type="float")
    parser.add_option("-a", "--average_limit", dest="average_limit", help="average_limit", default=0.9, metavar="AVERAGE_LIMIT", type="float")
    parser.add_option("-f", "--new_files", dest="new_files", help="new directory path", metavar="NEW_FILES")
    parser.add_option("-c", "--charts", dest="charts", help="charts directory path", default=defult_charts, metavar="CHARTS")
    return parser.parse_args()


if __name__ == '__main__':
    global array_average_percent
    global min_acc_array
    array_average_percent = []
    min_acc_array = []
    min_acc_array_positive = []
    min_acc_array_training = []
    options, _ = ApplyOptions("gripping_dynamic")
    print start(options.positives, options.experiments, options.test_size, options.average_limit, options.negatives, options.new_files)


