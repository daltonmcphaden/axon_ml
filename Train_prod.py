# Necessary libraries/tools
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from datetime import datetime
from Helpers import merge_linear_rotational
from Helpers import normalize_min_max
from Helpers import dbSensorData_to_dfs
from Helpers import femData_to_df
# from Helpers import visualize_loss
from Helpers import windowDf
import tensorflow as tf
from tensorflow import keras

import mysql.connector as mysql

# model_id = "test_12345"
# epochs = 50
# learning_rate = 0.001
# slidingWindow = "sliding"
# modelType = "integrated"
# caseIDs = "1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,20,21,22,23,24"

def train_model(model_id, epochs, learning_rate, slidingWindow, modelType, caseIDs):
    try:
        # enter server domain name
        HOST = "~~~"
        # database name
        DATABASE = "~~~"
        # user created
        USER = "~~~"
        # user password
        PASSWORD = "~~~"
        # port
        PORT = 25060
        # connect to MySQL server
        db_connection = mysql.connect(host=HOST, database=DATABASE, user=USER, password=PASSWORD, port=PORT)
        print("Connected to:", db_connection.get_server_info())

        caseID_str = "(" + caseIDs + ")"

        mysql_query_lin = "SELECT * FROM linearData WHERE ImpactID IN " + caseID_str
        mysql_query_rot = "SELECT * FROM rotationalData WHERE ImpactID IN " + caseID_str 
        
        mysql_query_fem = "SELECT * FROM fem WHERE ImpactID IN " + caseID_str   

        cursor = db_connection.cursor()

        cursor.execute(mysql_query_lin)
        linData = cursor.fetchall()    # get all selected rows        
        linDfs = dbSensorData_to_dfs(linData, True)                

        cursor.execute(mysql_query_rot)
        rotData = cursor.fetchall()
        rotDfs = dbSensorData_to_dfs(rotData, False)   

        cursor.execute(mysql_query_fem) 
        femData = cursor.fetchall()
        femDataDf = femData_to_df(femData)

    except mysql.Error as error:
        print("Failed: ", error)

    # Normalize the outputs
    # df_outputs[["Corpus Callosum", "Thalamus", "Brain Stem"]] = normalize_min_max(df_outputs[["Corpus Callosum", "Thalamus", "Brain Stem"]])

    data = []
    i=0
    print("Preparing data")    

    while i < len(linDfs):
        # case number
        case = linDfs[i]['caseID'][0]        

        # merged input timeseries
        timeseries_df = merge_linear_rotational(linDfs[i], rotDfs[i])         

        # outputs for the corresponding case number
        outputs = femDataDf.loc[femDataDf['Case'] == case]

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
        i+=1

    print("Data preparation complete")

    # filter for any null values in inputs
    print("Filter for nulls before normalization:")
    for d in data:
        # drop impact time column
        if d["inputs"].isnull().values.any():
            print("Case", d["case"], "removed because it has null values")
            data.remove(d)


    # normalize data
    maxLim = len(data)
    i = 0
    while i < maxLim: 
        if slidingWindow == "sliding":
            shifts = [-5, -3, -1, 1, 3, 5]
            for shift in shifts:
                df_clone = data[i]["inputs"].copy()
                df_clone = windowDf(df_clone,shift)              
                df_clone = normalize_min_max(df_clone)    

                data.append(
                    {
                        "case": data[i]["case"], 
                        "inputs": df_clone, 
                        "outputs": [
                            data[i]["outputs"][0],
                            data[i]["outputs"][1],
                            data[i]["outputs"][2]
                        ]
                    }
                )    

            data[i]["inputs"] = windowDf(data[i]["inputs"],0)     
            
        data[i]["inputs"] = normalize_min_max(data[i]["inputs"])    
        i += 1                        

    for d in data:
        d["inputs"] = np.asarray(d["inputs"]).astype(np.float32)

    train_split_fraction = 0.75
    trainThres = round(train_split_fraction * len(data))
    train_data = data[0: trainThres]
    val_data = data[trainThres:]

    train_inputs_array = np.array([np.array(row["inputs"]) for row in train_data])
    train_outputs_array = np.array([row["outputs"] for row in train_data])

    val_inputs_array = np.array([np.array(row["inputs"]) for row in val_data])
    val_outputs_array = np.array([row["outputs"] for row in val_data])

    print(train_inputs_array.dtype)
    print("train inputs shape:")
    print(train_inputs_array.shape)
    print("train outputs shape:")
    print(train_outputs_array.shape)
    print("val inputs shape:")
    print(val_inputs_array.shape)
    print("val outputs shape:")
    print(val_outputs_array.shape)

    inputs = keras.layers.Input(shape=(train_inputs_array.shape[1], train_inputs_array.shape[2]))
    lstm_out = keras.layers.LSTM(32)(inputs)    # densely connected NN with 32 dimension output space
    outputs = keras.layers.Dense(3, activation=keras.activations.relu)(lstm_out)   # 3 outputs (regions of the brain). 
    # By default, activation function is linear, but could apply relu

    # integrated refers to creating a model which predicts all 3 output variables simultaneously
    if modelType == "integrated":
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mape", metrics=['mse'])
        model.summary()
        
        history = model.fit(
            x=train_inputs_array,
            y=train_outputs_array,
            epochs=epochs,
            validation_data=(val_inputs_array, val_outputs_array)
        )

        # visualize_loss(history, "Training and Validation Loss")    

        # Save model        
        model.save('Machine Learning/saved_models/' + model_id + '.h5')
        print("Model:", model_id, "saved")

    # specialized refers to creating models which each predict a single output variable
    else:
        # break up output train data into separate arrays
        train_outputs_corpus_callosum = np.empty((len(train_outputs_array),1))
        train_outputs_thalamus = np.empty((len(train_outputs_array),1))
        train_outputs_brain_stem = np.empty((len(train_outputs_array),1))
        i=0
        for case in train_outputs_array:
            train_outputs_corpus_callosum[i] = case[0]
            train_outputs_thalamus[i] = case[1]
            train_outputs_brain_stem[i] = case[2]
            i+=1

        # break up output validation data into separate arrays 
        val_outputs_corpus_callosum = np.empty((len(val_outputs_array),1))
        val_outputs_thalamus = np.empty((len(val_outputs_array),1))
        val_outputs_brain_stem = np.empty((len(val_outputs_array),1))
        j=0
        for case in val_outputs_array:
            val_outputs_corpus_callosum[j] = case[0]
            val_outputs_thalamus[j] = case[1]
            val_outputs_brain_stem[j] = case[2]
            j+=1
            
        if modelType == "corpCall":
            model_single_cc = keras.Model(inputs=inputs, outputs=outputs)
            model_single_cc.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mape", metrics=['mse'])
            model_single_cc.summary()
            
            # Corpus Callosum
            history = model_single_cc.fit(
                x=train_inputs_array,
                y=train_outputs_corpus_callosum,
                epochs=epochs,
                validation_data=(val_inputs_array, val_outputs_corpus_callosum)
            )
            
            # visualize_loss(history, "Corpus Callosum Model: Training and Validation Loss")
        
            # Save model       
            model_id = model_id + "_CorpusCallosum"
            model_single_cc.save('saved_models/' + model_id + '.h5')
            print("Model:", model_id, " saved")
        
        elif modelType == "thalamus":
            model_single_th = keras.Model(inputs=inputs, outputs=outputs)
            model_single_th.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mape", metrics=['mse'])
            model_single_th.summary()

            # Thalamus
            history = model_single_th.fit(
                x=train_inputs_array,
                y=train_outputs_thalamus,
                epochs=epochs,
                validation_data=(val_inputs_array, val_outputs_thalamus)
            )

            # visualize_loss(history, "Thalamus Model: Training and Validation Loss")

            # Save model        
            model_id = model_id + "_Thalamus"
            model_single_th.save('saved_models/' + model_id + '.h5')
            print("Model:", model_id, " saved")
            
        else: # if modelType is brainStem
            model_single_bs = keras.Model(inputs=inputs, outputs=outputs)
            model_single_bs.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate), loss="mape", metrics=['mse'])
            model_single_bs.summary()

            # Brain Stem
            history = model_single_bs.fit(
                x=train_inputs_array,
                y=train_outputs_brain_stem,
                epochs=epochs,
                validation_data=(val_inputs_array, val_outputs_brain_stem)
            )

            # visualize_loss(history, "Brain Stem Model: Training and Validation Loss")

            # Save model        
            model_id = model_id + "_BrainStem"
            model_single_bs.save('saved_models/' + model_id + '.h5')
            print("Model:", model_id, " saved")        

    todayDate = datetime.today().strftime('%Y-%m-%d')

    mape = history.history["loss"]
    mse = history.history["mse"]
    mape_val = history.history["val_loss"]
    mse_val = history.history["val_mse"]

    ####### Build and execute insert sql statement to add the model details to model db #######
    # (modelID, createdBy, dateCreated, epoch, learning_rate, modelType, mape, mse, slidingWindow)
    modelDetailsUploadQuery = "INSERT INTO models VALUES ('" + model_id + "', '" + todayDate + "', " + str(epochs) + ", " + str(learning_rate) + ", '" + modelType + "', " + str(mape_val[-1]) + ", " + str(mse_val[-1]) + ", '" + slidingWindow + "');"

    try:
        cursor.execute(modelDetailsUploadQuery)
        db_connection.commit()    
    except:
        dupDeleteQuery = "DELETE FROM models WHERE modelID = '" + model_id + "';"
        cursor.execute(dupDeleteQuery)
        db_connection.commit()   
        
        dupDeleteQuery = "DELETE FROM metrics WHERE modelID = '" + model_id + "';"
        cursor.execute(dupDeleteQuery)
        db_connection.commit()   
        
        cursor.execute(modelDetailsUploadQuery)
        db_connection.commit()    

    i = 0
    thresholdVal = 0
    metricsQuery = "INSERT INTO metrics VALUES " 
    while i < epochs:
        metricsQuery += "('" + model_id + "', " + str(i+1) + ", " + str(mape[i]) + ", " + str(mape_val[i]) + ", " + str(mse[i]) + ", " + str(mse_val[i]) + "),"
        i += 1

        # break up the sql commands to 50 rows each
        if thresholdVal == 50 or i == epochs:
            metricsQuery = metricsQuery[:-1] #remove last comma
            metricsQuery += ";"                              
            cursor.execute(metricsQuery)
            db_connection.commit()
            thresholdVal = 0
            metricsQuery = "INSERT INTO metrics VALUES "             
        thresholdVal += 1      

    cursor.close()
    db_connection.close()

    return model_id