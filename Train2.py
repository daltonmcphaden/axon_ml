# Necessary libraries/tools
import pandas as pd
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE" # temp fix?
from Helpers import merge_linear_rotational2
from Helpers import normalize_min_max
from Helpers import visualize_loss
import tensorflow as tf
from tensorflow import keras

def train_model():
    inputs_folder = "Youth Soccer Head Impact Data"
    outputs_file = "Full_Brain_Region_Raw_Data.xlsx"
    df_outputs = pd.read_excel(outputs_file)

    data = []
    l = len(os.listdir(inputs_folder))
    i=0
    print("Preparing data")
    for filename in os.listdir(inputs_folder):
        i+=1
        # case number
        case = int(filename.split(".")[0])
        
        # merged input timeseries
        filepath = inputs_folder + '/' + filename
        input_data = pd.read_excel(filepath)
        timeseries_df = merge_linear_rotational2(filepath)
        
        # outputs for the corresponding case number
        outputs = df_outputs.loc[df_outputs['Case'] == case]
        
        # put it all together and append to an array
        data.append(
            {
                "case": case, 
                "inputs": timeseries_df, 
                "outputs": [
                    outputs["Corpus Callosum"].values[0],
                    outputs["Thalamus"].values[0],
                    outputs["Brain Stem"].values[0]
                ]
            }
        )    
    print("Data preparation complete")
    # filter for any null values in inputs
    print("Filter for nulls before normalization:")
    for d in data:
        # drop impact time column
        if d["inputs"].isnull().values.any():
            print("Case", d["case"], "removed because it has null values")
            data.remove(d)

    # normalize data
    for d in data:
        d["inputs"] = normalize_min_max(d["inputs"])

    # drop any nulls again after doing normalization
    print("Filter for nulls after normalization:")
    for d in data:
        # drop impact time column
        if d["inputs"].isnull().values.any():
            print("Case", d["case"], "removed because it has null values")
            data.remove(d)

    values = data[11]["inputs"].values

    learning_rate = 0.001
    train_split_fraction = 0.75
    epochs = 20

    trainThres = round(train_split_fraction * len(data))
    train_data = data[0: trainThres]
    val_data = data[trainThres:]

    train_inputs_array = np.array([np.array(row["inputs"]) for row in train_data])
    train_outputs_array = np.array([row["outputs"] for row in train_data])
    val_inputs_array = np.array([np.array(row["inputs"]) for row in val_data])
    val_outputs_array = np.array([row["outputs"] for row in val_data])

    inputs = keras.layers.Input(shape=(train_inputs_array.shape[1], train_inputs_array.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)    # densely connected NN with 32 dimension output space
    outputs = keras.layers.Dense(3)(lstm_out)   # 3 outputs (regions of the brain). 
    # By default, activation function is linear, but could apply relu or softmax

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mse")
    model.summary()

    history = model.fit(
        x=train_inputs_array,
        y=train_outputs_array,
        epochs=epochs,
        validation_data=(val_inputs_array, val_outputs_array)
    )

    # visualize_loss(history, "Training and Validation Loss")

    # Save model
    model_name = "test_001"
    model.save('saved_models/' + model_name + '.h5')
    print("Model:", model_name, "saved")