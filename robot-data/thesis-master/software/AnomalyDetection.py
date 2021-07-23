import RuleBasedAnomaly as RBA
import MachineLearning as ML
import Dataset as DS
import Measurements as M
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
      # pred = datasets_preds.intersection_labeling(pred, rba.transform_bit_same_column_values(type, dataset))
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

  datasets_preds.update_predictions(trainings_preds, positives_preds, negatives_preds)
  datasets_preds.set_datasets(training_datasets, positive_datasets, negative_datasets)
  return datasets_preds, str(rba)
  
def ML_stage(datasets_preds):
  trainings, positives, negatives = datasets_preds.get_datasets()
  ml = ML.MachineLearning(trainings, _nu = 0.01, _kernel = "rbf", _degree = 3, _gamma = 0.5)
  trainings_preds = ml.run_isolation_forest(*trainings)
  positives_preds = ml.run_isolation_forest(*positives)
  negatives_preds = ml.run_isolation_forest(*negatives)
  datasets_preds.update_predictions(trainings_preds, positives_preds, negatives_preds)
  return datasets_preds, str(ml)

def FS_stage(datasets_preds):
  import FeatureSelection as FS
  fs = FS.FeatureSelection()
  trainings, positives, negatives = datasets_preds.get_datasets()
  fs.fit(trainings)
  # new_trainings, new_positives, new_negatives = fs.variance_threshold(trainings), fs.variance_threshold(positives), fs.variance_threshold(negatives)
  # datasets_preds.set_datasets(new_trainings, new_positives, new_negatives)
  # new_trainings, new_positives,new_negatives = fs.pca(trainings), fs.pca(positives), fs.pca(negatives)
  # datasets_preds.set_datasets(new_trainings, new_positives, new_negatives)
  # new_trainings, new_positives, new_negatives = fs.select_k_best(trainings), fs.select_k_best(positives), fs.select_k_best(negatives)
  # datasets_preds.set_datasets(new_trainings, new_positives, new_negatives)
  return datasets_preds, ""

def start(positives_dir_path, negatives_dir_path):
  datasets_preds = DS.Datasets(positives_dir_path, negatives_dir_path, test_size=0.5, delete = 5)
  rba_info, fs_info, ml_info = "", "", ""
  datasets_preds, rba_info = RBA_stage(datasets_preds)
  # datasets_preds, fs_info = FS_stage(datasets_preds)
  # datasets_preds, ml_info = ML_stage(datasets_preds)
  return results(datasets_preds, rba_info = rba_info ,fs_info = fs_info, ml_info = ml_info)

an = 0
def predictions_information(datasets_preds):
  def get_acc_info(title, paths_preds, pred_value):
    global an
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
      warning = M.Measurements.count_warnings(y_pred,Conf.NEGATIVE_LABEL, options.threshold+1)
      if warning > 0:
        faulty_runs+=1
      warnings += warning
      tmp += "%s%s\taccuracy: %s%s\tlongest negative: %s\twarnings = %s\n" %(f, " " * (f_size - (len(f)-2)) , acc, " " * (acc_size - (len(str(acc))-2)),longest, warning)
      M.Measurements.draw_chart(f, options.charts + f[:-4],y_pred,Conf.NEGATIVE_LABEL)
    an = longests/len(paths_preds)
    ans = ans + "longest average = %s, anomalies = %s,  total warnings = %s,  faulty runs = %s,  non-faulty runs = %s\n" % (longests/len(paths_preds), anomalies, warnings, faulty_runs, len(paths_preds) - faulty_runs)
    return ans + tmp
  trainings_preds, positives_preds, negatives_preds = datasets_preds.get_predictions()
  trainings_names, positives_names, negatives_names = datasets_preds.get_names()
  ans = ""
  ans += get_acc_info("training sets", zip(trainings_names, trainings_preds), Conf.POSITIVE_LABEL) + "\n"
  ans += get_acc_info("positive sets", zip(positives_names, positives_preds), Conf.POSITIVE_LABEL) + "\n"
  ans += get_acc_info("negative sets", zip(negatives_names, negatives_preds), Conf.POSITIVE_LABEL) + "\n"
  return ans


def results(datasets_preds, rba_info ,fs_info, ml_info):
  info = predictions_information(datasets_preds)
  s = str(datasets_preds) + "\n"
  s += rba_info + "\n"
  s += fs_info + "\n"
  s += ml_info+ "\n"
  s += info + "\n"
  return s
#
# def get_files_name(path):
#   import glob
#   files = glob.glob(path + "*")
#   return [f +"/" for f in files]

def ApplyOptions(run):
  from optparse import OptionParser
  defult_positives = ""#get_files_name("/home/matansar/thesis/available runs/manipulation scenarios/%s/normal/" % (run,))
  defult_negatives = "/home/matansar/thesis/available runs/manipulation scenarios/%s/test/" % (run,)
  defult_charts = "/home/matansar/thesis/available runs/manipulation scenarios/%s/charts/" % (run,)
  parser = OptionParser()
  parser.add_option("-p", "--positives", dest="positives", help="positives directory path", default = defult_positives, metavar="POSITIVES")
  parser.add_option("-n", "--negatives", dest="negatives", help="negatives directory path", default = defult_negatives, metavar="NEGATIVES")
  parser.add_option("-c", "--charts", dest="charts", help="charts directory path", default = defult_charts, metavar="CHARTS")
  parser.add_option("-t", "--threshold", dest="threshold", help="threshold", default = 3, metavar="THRESHOLD", type="int")
  return parser.parse_args()

if __name__ == '__main__':
  options, _ = ApplyOptions("gripping_dynamic")
  results = start(options.positives, options.negatives)
  print results
