from sklearn import preprocessing 
import numpy as np
X = np.array([[ 1., -1.,  10.],
              [ 2.,  0.,  0.],
              [ 0.,  1., -1.]])
min_max_scaler = preprocessing.MinMaxScaler()
X_minMax = min_max_scaler.fit_transform(X)
print(X_minMax)