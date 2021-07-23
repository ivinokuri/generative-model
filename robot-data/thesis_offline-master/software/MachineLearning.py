import pandas as pd
from sklearn import svm
from Style import Style
from sklearn.ensemble import IsolationForest

csv_suffix = ".csv"
#import numpy as np
#options = ""
#info = ""


class MachineLearning:

  def __init__(self, trainings, _nu = 0.01, _kernel = "rbf", _degree = 3, _gamma = "auto"):
    self.training = pd.concat(trainings, ignore_index=True)
    self.nu = _nu
    self.kernel = _kernel
    self.degree = _degree
    self.gamma = _gamma
    self.forest = IsolationForest()
    self.forest.fit(self.training)
    self.svm = svm.OneClassSVM(nu = _nu, kernel = _kernel, degree = _degree, gamma = _gamma)
    self.svm.fit(self.training)
  
  def run_isolation_forest(self, *tests):
    tests_preds = []
    for test in tests:
      tests_preds.append(self.forest.predict(test))
    return tests_preds

  def run_one_class_svm(self, *tests):
    tests_preds = []
    for test in tests:
      tests_preds.append(self.svm.predict(test))
    return tests_preds

  def __str__(self):
    s = Style.BOLD + "Machine-Learning Detection:\n" + Style.END
    s += "svm params: nu = %s, kernel = %s, degree = %s, gamma = %s" % (self.nu, self.kernel, self.degree, self.gamma)
    return s

  # def __get_files_name(self, path):
  # import glob
  # files = []
  # if path[-len(csv_suffix):] == csv_suffix:
  # files = [path]
  # else:
  # files = glob.glob(path + ("*" + csv_suffix))
  # return files

  # def __get_machine_learning_method(self, method):
  # if method == 'one class svm':
  # algorithm = self.__run_one_class_svm
  # else:
  # algorithm = None
  # return algorithm

  # def set_one_class_svm_param(self, nu, kernel, degree, gamma):
  # self.svm_params = dict(_nu = nu, _kernel = kernel, _degree = degree, _gamma = gamma)

# select_fpr doesn't work


#def train_test_split_validation(ml_algorithm, positive_sets, negative_sets, _test_size = 0.3, times = 10):
  #global info
  #from sklearn.model_selection import train_test_split
  #info = "train-test-split validation = %s\n%s" % (_test_size , info)
  #summary = open(options.output, "w")
  ##negative_sets = map(lambda x: pd.read_csv(x, header=0), get_files(options.negative))
  ##attacker_sets = map(lambda x: pd.read_csv(x, header=0), get_files(options.attacker)) 
  ##positive_data = map(lambda x: pd.read_csv(x, header=0), get_files(options.train) + get_files(options.positive))
  #while times > 0:
    #training_sets, positive_test_sets = train_test_split(positive_sets, test_size= _test_size)
    #training_set = pd.concat(training_sets)
    #training_predictions, positive_predictions, negative_predictions, attacker_predictions = ml_algorithm(training_set, training_sets, positive_test_sets, negative_sets)
    #write_summary(summary, training_predictions, positive_predictions, negative_predictions, attacker_predictions)
    #summary.write("------------------------------------------------------------------------------------------------------------------------------\n")
    #summary.write("------------------------------------------------------------------------------------------------------------------------------\n")
    #summary_results(summary, [], [], training_predictions, positive_predictions, negative_predictions, attacker_predictions)
    #times = times - 1
  #summary.close()

#def k_cross_validation(ml_algorithm, _n_splits = 3):
  #global info
  #from sklearn.model_selection import KFold
  #info = "k-cross validation = %s\n%s" % (_n_splits, info)
  #summary = open(options.output, "w")
  #negs = get_files(options.negative)
  #negative_sets = map(lambda x: pd.read_csv(x, header=0),  negs) 
  #positive_data = map(lambda x: pd.read_csv(x, header=0), get_files(options.train) + get_files(options.positive))# + get_files(options.extra))  
  ##extra_sets = map(lambda x: pd.read_csv(x, header=0), get_files(options.extra)) 
  ##print extra_sets
  ##negative_sets, positive_data = history_notation(negative_sets, positive_data)
  #kf = KFold(n_splits = _n_splits, shuffle = True)
  #for train_index, test_index in kf.split(positive_data):
    #training_sets, positive_sets = split_array(positive_data, train_index, test_index)
    ##positive_sets = positive_sets + extra_sets
    #training_set = pd.concat(training_sets)
    #training_predictions, positive_predictions, negative_predictions = ml_algorithm(training_set, training_sets, positive_sets, negative_sets)
    ##write_summary(summary, training_predictions, positive_predictions, negative_predictions)
    #summary.write("------------------------------------------------------------------------------------------------------------------------------\n")
    #summary.write("------------------------------------------------------------------------------------------------------------------------------\n")
    #summary_results(summary, train_index, test_index, training_predictions, positive_predictions, negative_predictions)
    ##break
  #summary.close()
    
#def run_simple_algorithm(ml_algorithm):
  #training_set = get_data(get_files(options.train))
  #training_csv_files = map(lambda x: pd.read_csv(x, header=0), get_files(options.train)) #get_data(get_files(options.train))
  #positive_csv_files = map(lambda x: pd.read_csv(x, header=0), get_files(options.positive))
  #negative_csv_files = map(lambda x: pd.read_csv(x, header=0), get_files(options.negative))
  #training_predictions, positive_predictions, negative_predictions = ml_algorithm(training_set, training_csv_files, positive_csv_files, negative_csv_files)
  #write_results(training_predictions, positive_predictions, negative_predictions)


# --------------------------------------------------------------------- history notation ---------------------------------------------------------------------
#def history_notation(negative_sets, positive_sets):
  #new_negative_sets = map(lambda x: produce_history(x),negative_sets)
  #new_positive_sets = map(lambda x: produce_history(x),positive_sets)
  #return new_negative_sets, new_positive_sets

#history_counter = 0
#def produce_history(df, alpha = 0.1):
  #global info, history_counter
  #if history_counter == 0:
    #info = "%shistory notation: %s\n" % (info,alpha)
  #titles = df.columns
  #dataset = df.as_matrix()
  #new_dataset = []
  #last = dataset[0]
  #new_dataset.append(last)
  #for features in dataset[1:]:
    #new_features = create_new_features(last, features, alpha)
    #new_dataset.append(new_features)
    #last = new_features
  #history_counter = history_counter + 1
  #return pd.DataFrame(data = new_dataset, columns = titles)
  
#def create_new_features(acc, curr, alpha):
  #ret = []  
  #for i in range(0,len(acc)):
    #x = (1-alpha)*curr[i] + alpha*acc[i]
    #ret.append(x)
  #return ret
  

# --------------------------------------------------------------------- help function---------------------------------------------------------------------

#result_counter = 0
#def summary_results(f, train_index, test_index, training_predictions, positive_predictions, negative_predictions, attacker_predictions = []):
  #global result_counter
  #if result_counter == 0:
    #f.write(info)
  #training_seq = map(lambda x: longest_sequence(x, FALSE_LABEL), training_predictions)
  #negative_seq = map(lambda x: longest_sequence(x, FALSE_LABEL), positive_predictions)
  #positive_seq = map(lambda x: longest_sequence(x, FALSE_LABEL), negative_predictions)
  
  #training_cnt = map(lambda x: counter(x), training_predictions)
  #negative_cnt = map(lambda x: counter(x), positive_predictions)
  #positive_cnt = map(lambda x: counter(x), negative_predictions)
  
  #title = "\t\t\t\t\t\t\t-----------------------result N.%s-----------------------\n" % (result_counter)
  #indexes = "files for training: %s\nfiles for testing: %s\n" % (train_index, test_index)
  #training_result = "%s longest sequence of %s: %s, max = %s, count 1: %s\n" % ("training-set", FALSE_LABEL, training_seq, max(training_seq), training_cnt)
  #positive_result = "%s longest sequence of %s: %s, max = %s, count 1: %s\n" % ("positive-set", FALSE_LABEL, negative_seq, max(negative_seq), positive_cnt)
  #negative_result = "%s longest sequence of %s: %s, min = %s, count 1: %s\n\n\n" % ("negative-set", FALSE_LABEL, positive_seq, min(positive_seq), negative_cnt)
  #f.write(title)
  #f.write(indexes)
  #f.write(training_result)
  #f.write(positive_result)
  #f.write(negative_result)
  #result_counter = result_counter + 1
 
    

#def counter(array, value = TRUE_LABEL):
  #counter = 0.0
  #for i in array:
    #if i == value:
      #counter = counter + 1
  #return counter/len(array)
    

#def longest_sequence(array, value = TRUE_LABEL):
  #longest = 0
  #for i in range(0, len(array)):
    #if array[i] == value:
      #counter = 1
      #for j in range(i+1, len(array)):
	#if array[i] == array[j]:
	  #counter = counter + 1
	#else:
	  #break
      #longest = max(longest, counter)
  #return longest


# --------------------------------------------------------------------- generic functions ---------------------------------------------------------------------

#def write_summary(f, training_prediction, positive_prediction, negative_prediction, attacker_prediction = []):
  #def print_2d_array(f, array, value):
    #for i in array:
      #counter = 0
      ##f.write(str(i)+ "\n")
      #for j in i:
	#if j == value:
	  #counter = counter +1
      #f.write("summary value=%s: %s/%s=  \t%s\n" % (value, counter, len(i), round(float(counter)/len(i),9)))
  #f.write(info + "\n\n")
  #f.write("training_prediction =\n")
  #print_2d_array(f, training_prediction, TRUE_LABEL)
  ##f.write("training_prediction = " + str(training_prediction)+ "\n")
  #f.write("\n\npositive_prediction =\n")
  #print_2d_array(f, positive_prediction, TRUE_LABEL)
  #f.write("\n\nnegative_prediction =\n")
  #print_2d_array(f, negative_prediction, TRUE_LABEL)
  #f.write("\n\nattacker_prediction =\n")
  #print_2d_array(f, attacker_prediction, TRUE_LABEL)

#def write_results(training_prediction, positive_prediction, negative_prediction):
  #def print_2d_array(f, array, value):
    #for i in array:
      #counter = 0
      #f.write(str(i)+ "\n")
      #for j in i:
	#if j == value:
	  #counter = counter +1
      #f.write("summary value=%s: %s/%s=%s\n\n" % (value, counter, len(i), float(counter)/len(i)))
  #with open(options.output, "w") as f: 
    #f.write(info + "\n\n")
    #f.write("training_prediction =\n")
    #print_2d_array(f, training_prediction, TRUE_LABEL)
    ##f.write("training_prediction = " + str(training_prediction)+ "\n")
    #f.write("\n\npositive_prediction =\n")
    #print_2d_array(f, positive_prediction, TRUE_LABEL)
    #f.write("\n\nnegative_prediction =\n")
    #print_2d_array(f, negative_prediction, TRUE_LABEL)
      
#def calculate_measurements(true_labels, pred_labels):
  #fpr = calculate_FP(true_labels, pred_labels)
  #tpr = calculate_TPR(true_labels, pred_labels)
  #tnr = calculate_TNR(true_labels, pred_labels)
  #ppv = calculate_PPV(true_labels, pred_labels)
  #return fpr,tpr,tnr,ppv
  ##print "FPR = ", fpr
  ##print "TPR = ", tpr
  ##print "TNR = ", tnr
  ##print "PPV = ", ppv

#def check_appriciate_columns(dfs):
  #for (df1,df2) in zip(dfs[:-1],dfs[1:]):
    #if len(df1.columns) != len(df2.columns):
      #print "Number of columns is not suitable"

#def get_data(csv_files):
  #dfs = map(lambda x: pd.read_csv(x, header=0), csv_files)
  #check_appriciate_columns(dfs)
  #concate = pd.concat(dfs)
  #return concate

#def ApplyOptions():
  #global options
  #from optparse import OptionParser
  #parser = OptionParser()
  #parser.add_option("-e", "--extract", dest="extract", help="extract", default = "1", metavar="EXTRACT")
  #parser.add_option("-p", "--positive", dest="positive", help="positive set path", default = "", metavar="POSITIVE")
  #parser.add_option("-n", "--negative", dest="negative", help="negative set path", default = "", metavar="NEGATIVE")
  #parser.add_option("-o", "--output", dest="output", help="write report to OUT (.txt)", default = "result.txt", metavar="OUT")
  
  #parser.add_option("-t", "--train", dest="train", help="training set path", default = "", metavar="TRAIN")
  #parser.add_option("-w", "--window", dest="window", help="window", default = "5", metavar="WINDOW")
  #parser.add_option("-v", "--version", dest="version", help="version", default = "hz_bw", metavar="VERSION")
  #parser.add_option("-f", "--features", dest="features", help="features", default = "map", metavar="FEATURES")
  #parser.add_option("-s", "--step", dest="step", help="step", default = "1", metavar="STEP")
  
  #parser.add_option("-a", "--attacker", dest="attacker", help="attacker", default = "", metavar="ATTACKER")
  #options, args  = parser.parse_args()
  

#if __name__ == "__main__":
  #ApplyOptions()
  #options.positive = "/home/matansar/MEGA/ros simulations/normal behavior/csv/"
  #options.negative = "/home/matansar/MEGA/ros simulations/nav_vel publising attacking/rate=4/csv"
  ##positive_sets = map(lambda x: pd.read_csv(x, header=0),  (get_files_name(options.positive)))  
  ##negative_sets = map(lambda x: pd.read_csv(x, header=0),  (get_files_name(options.negative)))
  
  #anomaly = AnomalyDetection(options.positive, options.negative)
  ##anomaly.apply_train_test_split(method = 'one class svm')
  #anomaly.apply_k_cross_validation(method = 'one class svm')
  ##get_data(get_files(options.negative))
  
  ##options.output = "r_v%s_ts_%s_s_%s" % (options.version, options.window, options.step)
  ##train_test_split_validation(run_one_class_svm, get_files_name(options.positive), get_files_name(options.negative))
  ###k_cross_validation(run_one_class_svm)
    
    
    
### sheet function

#def check_variante(training_csv_files):
  #array = training_csv_files.as_matrix()
  #counter = 0
  #for column in range(0,len(array[0])):
    #flag = True
    #value = array[0][column]
    #for row in range(0,len(array)):
      #if array[row][column] != value:
	#flag = False
	#break
    #if flag == True:
      #counter = counter + 1
  #print len(array[0]) - counter
    
  