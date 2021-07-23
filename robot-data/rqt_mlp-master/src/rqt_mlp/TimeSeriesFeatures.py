#phase 3

from decimal import Decimal
import pandas as pd
import statistics

options = None

UniFeatures = "Univariate Features"
MultiFeatures = "Multivariate Features"

class TimeSeries:

# ------------------------------------------------------------ constructor ------------------------------------------------------------  
  
  def __init__(self, input_path, output_path, window, time_series_features_selection, step = 1):
    self.__input_path = input_path
    self.__output_path = output_path
    self.__window = window
    self.__step = step
    self.__time_series_features_name = []
    #print time_series_features_selection[UniFeatures]
    self.set_univariate_features_selection(time_series_features_selection[UniFeatures])
    self.set_multivariate_features_selection(time_series_features_selection[MultiFeatures])


# ------------------------------------------------------------ setters ------------------------------------------------------------  

  
  def set_univariate_features_selection(self, univariate_features_selection):
    value_index = 1
    #univariate_features_selection = sorted(univariate_features_selection)
    if (set(univariate_features_selection) <= set(get_time_series_pre_feature_options()[value_index])):
      self.__univariate_features_selection = sorted(list(univariate_features_selection)) # new list 
    else:
      self.__univariate_features_selection = []
      print('ERROR: the time series pre feature selection is not a subset of time series features options')


  def set_multivariate_features_selection(self, multivariate_features_selection):
    value_index = 1
    #multivariate_features_selection = sorted(multivariate_features_selection)
    if (set(multivariate_features_selection) <= set(get_global_time_series_features_options()[value_index])):
      self.__multivariate_features_selection = sorted(list(multivariate_features_selection)) # new list 
    else:
      self.__multivariate_features_selection = []
      print('ERROR: the global time series features selection is not a subset of time series features options')



# ------------------------------------------------------------ functionality ------------------------------------------------------------  


  def generate_time_series_features(self):
    df = pd.DataFrame()
    features_names, dataset = TimeSeries.__read_data_set(self.__input_path)
    l = len(dataset)
    if l >= self.__window:
      for i in range(0, l - self.__window + 1, self.__step):
	jumping = i + self.__window
	time_series_features = self.__generate_time_series_feature(features_names, dataset[i: jumping])
	df = df.append([time_series_features])
    else:
      print('ERROR: window size is too big for this dataset')
      print('\t\twindow size = %s | dataset length = %s' % (self.__window, l))
    df.columns = self.__time_series_features_name
    df.to_csv(self.__output_path, index=False)

# ------------------------------------------------------------ private functions ------------------------------------------------------------  
  
  
  def __update_features_name(self, *names):
    for name in names:
      if name not in self.__time_series_features_name:
	self.__time_series_features_name.append(name)

  def __generate_time_series_feature(self, features_names, dataset):
    new_features = []
    l = len(dataset)
    features_counter = len(dataset[0])
    # time series pre feature
    for feature_i in range(0, features_counter):
      feature_name = features_names[feature_i]
      feature_values = map(lambda x: x[feature_i], dataset) 
      for ts_feature in self.__univariate_features_selection:
	self.__update_features_name("%s(%s)"% (ts_feature, feature_name))
	function = TimeSeries.__get_related_funtion(ts_feature)
	result = function(feature_values)
	new_features.append(result)
    # time series global dataset
    for ts_feature in self.__multivariate_features_selection:
      self.__update_features_name(ts_feature)
      function = TimeSeries.__get_related_funtion(ts_feature)
      result = function(features_names, dataset)
      new_features.append(result)
    return new_features
  
  # the dataset has a header
  @staticmethod
  def __read_data_set(dataset_path):
    df = pd.read_csv(dataset_path, header=0)
    return list(df.columns), df.as_matrix()
  
  @staticmethod
  def __get_related_funtion(selection):
    try:
      function = time_series_per_feature_options[selection]
    except KeyError:
      function = global_time_series_features_options[selection]
    return function
  
  
# ----------------------------------------------------------------- features ------------------------------------------------------------  

def last(points):
  last = points[-1]
  return last

def df(points):
  first = points[0]
  last = points[-1]
  return (last - first)/1.0

def ddf(points):
  l = len(points)
  first = points[0]
  middle = points[l/2]
  last = points[-1]
  first_half = df([first,middle])
  second_half = df([middle,last])
  return df([first_half, second_half])

def distance(features_names, dataset):
  import math
  try:
    location_x = features_names.index('location x')
    location_y = features_names.index('location y')
    first = dataset[0]
    last = dataset[-1]
    dx = last[location_x] - first[location_x]
    dy = last[location_y] - first[location_y]
    distance = math.sqrt(pow(dx,2) + pow(dy,2))
  except ValueError:
    distance = 0
  return distance

def average(values):
    s = sum(values) + 0.0
    return s/len(values)


# ----------------------------------------------------------------- global variables ------------------------------------------------------------  


time_series_per_feature_options = { 'average' : average, 'last occurrence' : last , 'first derivative' : df, 'second derivative' : ddf}
global_time_series_features_options = {'passed distance' : distance}

# ----------------------------------------------------------------- getters ------------------------------------------------------------  

def get_time_series_pre_feature_options():
  return UniFeatures, time_series_per_feature_options.keys()

def get_global_time_series_features_options():
  return MultiFeatures, global_time_series_features_options.keys()



## --------------------------------------------------------------------------------------------------------------------------------------------
## --------------------------------------------------------------------------------------------------------------------------------------------
## ---------------------------------------------------- main ----------------------------------------------------------------------------------



def write_data_set(path_file, df):
  df.to_csv(path_file, index=False)
 
def ApplyOptions():
    from optparse import OptionParser
    global options
    parser = OptionParser()
    parser.add_option("-i", "--input", dest="input",help = "", metavar="IN")
    parser.add_option("-o", "--output", dest="output",help = "", metavar="OUT")
    options, args  = parser.parse_args()  
  
#if __name__ == "__main__": 
  #ApplyOptions()
  #key1, value1 = get_time_series_pre_feature_options()
  #key2, value2 = get_global_time_series_features_options()
  #time_series_features_selection = dict()
  #time_series_features_selection[key1] = value1
  #time_series_features_selection[key2] = value2
  #window = 5
  #ts = TimeSeries(options.input, options.output, window, time_series_features_selection)
  #ts.generate_time_series_features()
  
  ##df = ts.generate_time_series_features()
  ##write_data_set(options.output, df)
  ##write_data_set
  ##print df
 
