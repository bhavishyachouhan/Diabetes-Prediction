

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import pickle
#loading the saved model
loaded_model=pickle.load(open('"C:/Users\HP/Downloads/ml project/trained_model (1).sav"','rb'))
input_data=(5,166,72,19,175,25.8,0.587,51)
#changing the input_data to numpy array
input_data_as_numpy_array=np.asarray(input_data)
#reshaping the array as we are predicting for only one instance or record
input_data_reshaped=input_data_as_numpy_array.reshape(1,-1)
scaler =StandardScaler()
input_standardized=scaler.fit_transform(input_data_reshaped)
prediction=loaded_model.predict(input_standardized)
if(prediction[0]==0):
  print('person is non diabetic')
else:
  print('person is diabetic')