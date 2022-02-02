### Structure

 - data-adaptation - this folder contains script for data preprocessing and columns filtering
 - py - all the sources of python code including notebooks
   - model 
     - autoencoder - source code of LSTM autoencoder
     - stan - all the notebooks of experiments with Stan
   - sim - not relevant
 - robot-data
   - new_data - data received from Ronens student
     - normal - normal data of 3 different simulations
     - test - data contains anomaly for different simulations
 - src - old Julia code, not relevant


### Installations 
For running notebooks need to install next packages:
 - pystan
 - pandas
 - scipy
 - heapq
 - tqdm
 - sklearn
 - pickle
 - keras
 - tensorflow

### Running 

All the experiments with the data are in py/model/stan directory

 - 2D-GMM_pick_miss_cup - Multidimentional GMM with data of picking cup (Autoencoder and GMM)
 - ProbabilisticModelExperiments - Experiments with old data, not relevant 
 - ProbabilisticModelExperiments_build - Experiments with new data of moving robot (PCA and GMM)
 - ProbabilisticModelExperiments_cans - Experiments with new data (PCA and GMM)
 - ProbabilisticModelExperiments_corr - Experiments with new data (PCA and GMM)
 - ProbabilisticModelExperiments_pick_miss_cup - Experiments with new data (PCA and GMM)
 - StanAutoencoder_pick_miss_cup - GMM with data of picking cup experiments (Autoencoder and GMM)