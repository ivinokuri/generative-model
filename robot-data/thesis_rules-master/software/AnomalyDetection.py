import RuleBasedAnomaly as RBA
import Dataset as DS
import Measurements as M
import json
import inspect, os
from Style import Configure as Conf

global options


def RBA_stage(datasets_preds):
  def apply_rba(rba, type, datasets):
    datasets_predictions = []
    removable_cols = rba.get_removable_columns()
    new_datasets = []
    for dataset in datasets:
      pred = rba.transform_same_digits_number(type, dataset)
      pred = datasets_preds.intersection_labeling(pred, rba.transform_positive_values(type, dataset))
      pred = datasets_preds.intersection_labeling(pred, rba.transform_negative_values(type, dataset))
      pred = datasets_preds.intersection_labeling(pred, rba.transform_not_negative_values(type, dataset))
      pred = datasets_preds.intersection_labeling(pred, rba.transform_coverage_percentage_columns(type, dataset))
      pred = datasets_preds.intersection_labeling(pred, rba.transform_corresponding_columns(type, dataset))
      pred = datasets_preds.intersection_labeling(pred, rba.transform_exactly_one_value(type, dataset))
      datasets_predictions.append(pred)
      new_datasets.append(DS.Datasets.remove_columns(dataset, removable_cols))

    return new_datasets, datasets_predictions

  trainings, positives, negatives = datasets_preds.get_datasets()
  rba = RBA.RuleBasedAnomaly()

  rba.fit(trainings, percentage=0.01, coverage=0.98)
  # rba.fit(trainings, percentage=0.05, coverage=0.97)
  # rba.fit(trainings, percentage = 0.001, coverage = 0.99)
  training_datasets, trainings_preds = apply_rba(rba, Conf.TRAINING, trainings)
  positive_datasets, positives_preds = apply_rba(rba, Conf.POSITIVE, positives)
  negative_datasets, negatives_preds = apply_rba(rba, Conf.NEGATIVE, negatives)

  # save_rules_learning_to_file(rba.rules_stats, rba.deleted_topics)
  new_save_rules_learning_to_file(rba.rules_fit)
  datasets_preds.update_predictions(trainings_preds, positives_preds, negatives_preds)
  datasets_preds.set_datasets(training_datasets, positive_datasets, negative_datasets)
  return datasets_preds, str(rba)

# def save_rules_learning_to_file(rules_stats, deleted_topics):
#     filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/rules.json"
#     for topic in deleted_topics:
#         for rule in topic:
#             if topic in rules_stats and rule in rules_stats[topic]:
#                 del(rules_stats[topic][rule])
#     f = open(filepath, "w")
#     json.dump(rules_stats, f)

def new_save_rules_learning_to_file(rules_stats):
    filepath = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe()))) + "/rules.json"
    f = open(filepath, "w")
    json.dump(rules_stats, f)


# def start(positives_dir_path, negatives_dir_path, trainings_dir_path):
#   datasets_preds = DS.Datasets(positives_dir_path, negatives_dir_path, trainings_dir_path, test_size=0.5, delete = 5)

def start(positives_dir_path, negatives_dir_path):
  print "starting"
  datasets_preds = DS.Datasets(positives_dir_path, negatives_dir_path, test_size=0, delete=1) #int(4 / float(0.25)))
  datasets_preds, rba_info = RBA_stage(datasets_preds)
  return results(datasets_preds, rba_info = rba_info)


# an = 0
def predictions_information(datasets_preds):
  def get_acc_info(title, paths_preds, pred_value, max_longest):
    # global an
    ans = "------------------------ %s ------------------------\n" % (title,)
    tmp = ""
    longests = 0.0
    anomalies = 0.0
    warnings = 0.0
    faulty_runs = 0.0
    acc_size = 5
    f_size = 30
    for f, y_pred in paths_preds:
      y_true = [pred_value] * len(y_pred)
      acc = round(M.Measurements.accuracy_score(y_true, y_pred), acc_size)
      anomalies += len(filter(lambda x: x != pred_value, y_pred))
      longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
      longests += longest
      warning = M.Measurements.count_warnings(y_pred, Conf.NEGATIVE_LABEL, int(max_longest + 1))
      #warning = M.Measurements.count_warnings(y_pred,Conf.NEGATIVE_LABEL, options.threshold+1)
      if warning > 0:
        faulty_runs+=1
      warnings += warning
      tmp += "%s%s\taccuracy: %s%s\tlongest negative: %s\twarnings = %s\n" %(f, " " * (f_size - (len(f)-2)) , acc, " " * (acc_size - (len(str(acc))-2)),longest, warning)
      M.Measurements.draw_chart(f, options.charts + f[:-4],y_pred,Conf.NEGATIVE_LABEL)
    # an = longests/len(paths_preds)
    ans = ans + "longest average = %s, anomalies = %s,  total warnings = %s,  faulty runs = %s,  non-faulty runs = %s\n" % (longests/len(paths_preds), anomalies, warnings, faulty_runs, len(paths_preds) - faulty_runs)
    return ans + tmp
  trainings_preds, positives_preds, negatives_preds = datasets_preds.get_predictions()
  trainings_names, positives_names, negatives_names = datasets_preds.get_names()
  ans = ""
  max_longest = 0.0
  for f, y_pred in zip(trainings_names, trainings_preds):
      longest = M.Measurements.longest_sequence(y_pred, Conf.NEGATIVE_LABEL)
      if longest > max_longest:
          max_longest = longest
  print "max threshold " + str(max_longest)
  ans += get_acc_info("training sets", zip(trainings_names, trainings_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
  if positives_names:
    ans += get_acc_info("positive sets", zip(positives_names, positives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
  if negatives_names:
    ans += get_acc_info("negative sets", zip(negatives_names, negatives_preds), Conf.POSITIVE_LABEL, max_longest) + "\n"
  return ans


def results(datasets_preds, rba_info):
  info = predictions_information(datasets_preds)
  s = str(datasets_preds) + "\n"
  s += rba_info + "\n"
  s += info + "\n"
  return s


def ApplyOptions(run):
  from optparse import OptionParser
  defult_positives = ""#get_files_name("/home/matansar/thesis/available runs/manipulation scenarios/%s/normal/" % (run,))
  defult_negatives = "/home/matansar/thesis/available runs/manipulation scenarios/%s/test/" % (run,)
  defult_charts = "/home/matansar/thesis/available runs/manipulation scenarios/%s/charts/" % (run,)
  parser = OptionParser()
  parser.add_option("-p", "--positives", dest="positives", help="positives directory path", default=defult_positives, metavar="POSITIVES")
  parser.add_option("-n", "--negatives", dest="negatives", help="negatives directory path", default=defult_negatives, metavar="NEGATIVES")
  parser.add_option("-r", "--trainings", dest="trainings", help="trainings directory path", default=defult_negatives, metavar="TRANINGS")
  parser.add_option("-c", "--charts", dest="charts", help="charts directory path", default = defult_charts, metavar="CHARTS")
  parser.add_option("-t", "--threshold", dest="threshold", help="threshold", default=3, metavar="THRESHOLD", type="int")
  return parser.parse_args()


if __name__ == '__main__':
  options, _ = ApplyOptions("gripping_dynamic")
  results = start(options.positives, options.negatives)
  # results = start(options.positives, options.negatives, options.trainings)
  print results
