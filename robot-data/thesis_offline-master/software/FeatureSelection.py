from Style import Configure as Conf
import pandas as pd
class FeatureSelection:
  
  def __init__(self, k = 30, percentile = 2, threshold = 0, n_components = 50):
    self.training = None
    self.k = k
    self.percentile = percentile
    self.threshold = threshold
    self.n_components = n_components
    

  def fit(self, trainings):
    from sklearn.feature_selection import SelectKBest
    from sklearn.feature_selection import SelectPercentile
    from sklearn.feature_selection import VarianceThreshold
    from sklearn.decomposition import PCA
    training = pd.concat(trainings, ignore_index=True)
    self.training = training
    labels = [Conf.POSITIVE_LABEL] * len(training)
    self.k_best_model = SelectKBest(k = self.k)
    self.k_best_model.fit(training, labels)
    self.percentile_model = SelectPercentile(percentile = self.percentile)
    self.percentile_model.fit(training, labels)
    self.threshold_model = VarianceThreshold(threshold = self.threshold)
    self.threshold_model.fit(training, labels)
    self.pca_model = PCA(n_components= self.n_components)
    self.pca_model.fit(training, labels)
    print len(filter(lambda x: x==True, self.threshold_model.get_support()))
    
  
  def select_k_best(self, datasets): # univariate_feature_selection
    #global info
    #info = "%sfeature selection: %s, %s\n" % (info,"SelectKBest", _k)
    new_datasets = []
    for dataset in datasets:
      new_dataset =  pd.DataFrame(self.k_best_model.transform(dataset))
      new_datasets.append(new_dataset)
    return new_datasets
  

  def select_percentile(self, datasets): # univariate_feature_selection
    #global info
    #info = "%sfeature selection: %s, %s\n" % (info,"SelectPercentile", _percentile)
    new_datasets = []
    for dataset in datasets:
      new_dataset =  pd.DataFrame(self.percentile_model.transform(dataset))
      new_datasets.append(new_dataset)
    return new_datasets
  

  ### This feature selection algorithm looks only at the features (X), not the desired outputs (y),
  ### and can thus be used for unsupervised learning.
  ### Features with a training-set variance lower than this threshold will be removed.
  ### threshold = 0 removes the colums with the same value
  def variance_threshold(self, datasets):
    #global info
    #info = "%sfeature selection: %s, %s\n" % (info,"VarianceThreshold", _threshold)

    new_datasets = []
    for dataset in datasets:
      new_dataset =  pd.DataFrame(self.threshold_model.transform(dataset))
      new_datasets.append(new_dataset)
    return new_datasets
    
  def pca(self, datasets):
    print self.pca_model
    new_datasets = []
    for dataset in datasets:
      new_dataset = pd.DataFrame(self.pca_model.transform(dataset))
      new_datasets.append(new_dataset)
    return new_datasets
