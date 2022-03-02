import os
from sys import argv
from Helpers import merge_linear_rotational2
from Helpers import normalize_min_max
from tensorflow import keras
import numpy as np

def generate_predictions():
    model_name = "test_1056"
    model = keras.models.load_model('saved_models/' + model_name + '.h5')

    inputs_folder = "Youth Soccer Head Impact Data 2"
    timeseries_df_Acc = []
    for file in os.listdir(inputs_folder):
        path = inputs_folder + '/' + file
        print(path)
        timeseries_df = merge_linear_rotational2(path)
        timeseries_df = normalize_min_max(timeseries_df)
        timeseries_df_Acc.append(timeseries_df)    
    numOfInputs = len(timeseries_df_Acc)

    array = np.empty((numOfInputs,120,8))

    i = 0
    for timeseries_dfr in timeseries_df_Acc:
        ts_arr = timeseries_dfr.to_numpy()
        array[i] = ts_arr
        i = i + 1    

    return model.predict(array) 