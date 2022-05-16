from Helpers import merge_linear_rotational
from Helpers import normalize_min_max
from Helpers import dbSensorData_to_dfs
from Helpers import windowDf

from tensorflow import keras
import numpy as np
import mysql.connector as mysql
import math
import os

caseIDs = "1,2,3,4,5,6"
model_id = "test_12345"

def generate_predictions(batchName, caseIDs, model_id):     
    try:
        windowSize = 20

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

        cursor = db_connection.cursor()

        cursor.execute(mysql_query_lin)
        linData = cursor.fetchall()    # get all selected rows        
        linDfs = dbSensorData_to_dfs(linData, True)                

        cursor.execute(mysql_query_rot)
        rotData = cursor.fetchall()
        rotDfs = dbSensorData_to_dfs(rotData, False)    
        
        mysql_query_modelDetails = "SELECT * FROM models WHERE modelID = '" + model_id + "';"        
        cursor.execute(mysql_query_modelDetails)
        modelDetails = cursor.fetchall()[0]

        modelType = modelDetails[4]
        slidingWindow = modelDetails[7]
        
    except mysql.Error as error:
        print("Failed: ", error)

    timeseries_df_Acc = []
    i = 0
    caseIDCounter = 0
    caseIDs = caseIDs.split(",")
    # combine and normalize each of the respective linear and rotational sets
    while i < len(linDfs):
        timeseries_df = merge_linear_rotational(linDfs[i], rotDfs[i])  

        # filter for any null values in inputs                 
        if timeseries_df.isnull().values.any():
            print("Case", caseIDs[caseIDCounter],"removed because it has null values")
            del caseIDs[caseIDCounter]
            i += 1
            continue

        if slidingWindow == "sliding":
            timeseries_df = windowDf(timeseries_df,0)   
        
        timeseries_df = normalize_min_max(timeseries_df)    
        timeseries_df_Acc.append(timeseries_df)    
        i += 1
        caseIDCounter += 1
    numOfInputs = len(timeseries_df_Acc)

    # create an array containing the inputs to be predicted.
    # start with empty numpy array, but specify the shape
    # first param should is length of input batch size. If doing prediction for a single input, shape[0] is 1
    # second param (120) is number of rows (timesteps) and third param is number of features
    if slidingWindow == "sliding":
        array = np.empty((numOfInputs,windowSize*2+1,8))
    else:
        array = np.empty((numOfInputs,120,8))

    i = 0
    for timeseries_dfr in timeseries_df_Acc:        
        ts_arr = timeseries_dfr.to_numpy()
        array[i] = ts_arr
        i = i + 1    

    print("Current directory", os.getcwd()) 
    model = keras.models.load_model('Machine Learning/saved_models/' + model_id + '.h5')
    prediction = model.predict(array)    

    print("Predictions of Maximum Principal Strain:")
    if modelType == "integrated":
        for p in prediction:
            print("MPS in Corpus Callosum:", p[0])
            print("MPS in Thalamus:", p[1])
            print("MPS in Brain Stem:", p[2])
            print("")
    else:
        if modelType == "corpCall":
            regionName = "Corpus Callosum"
            for p in prediction:
                print("MPS Corpus Callosum:", p[0])
        elif modelType == "thalamus":
            regionName = "Thalamus"
            for p in prediction:
                print("MPS Thalamus:", p[0])
        else:
            regionName = "Brain Stem"
            for p in prediction:    
                print("MPS Brain Stem:", p[0])            


    ####### Build and execute insert sql statement to add these predictions to db #######
    predictUploadQuery = "INSERT INTO predictions VALUES " 

    i = 0
    thresholdVal = 0

    if modelType == "integrated":
        for p in prediction:
            # set nan predictions to be -100
            if math.isnan(p[0]):
                p[0] = -100
            if math.isnan(p[1]):
                p[1] = -100
            if math.isnan(p[2]):
                p[2] = -100

            # (batchName, impactID, outputType, OutputValue, modelID)
            predictUploadQuery += "('" + batchName + "', " + str(int(caseIDs[i])) + ",'Corpus Callosum'," + str(p[0]) + ",'" + str(model_id) +"'),"
            predictUploadQuery += "('" + batchName + "', " + str(int(caseIDs[i])) + ",'Thalamus'," + str(p[1]) + ",'" + str(model_id) +"'),"
            predictUploadQuery += "('" + batchName + "', " + str(int(caseIDs[i])) + ",'Brain Stem'," + str(p[2]) + ",'" + str(model_id) +"'),"
            
            i += 1
            
            # break up the sql commands to 30 rows each
            if thresholdVal == 30 or i == len(prediction):
                
                predictUploadQuery = predictUploadQuery[:-1] #remove last comma
                predictUploadQuery += ";"                      
                cursor.execute(predictUploadQuery)
                db_connection.commit()
                thresholdVal = 0
                predictUploadQuery = "INSERT INTO predictions VALUES "                                 
            thresholdVal += 1      

    else: # specialized model
        for p in prediction:
            # set nan predictions to be -100
            if math.isnan(p[0]):
                p[0] = -100

            # (batchName, impactID, outputType, OutputValue, modelID)
            predictUploadQuery += "('" + batchName + "', " + str(caseIDs[i]) + ",'" + regionName + "'," + str(p[0]) + ",'" + str(model_id) + "'),"

            i += 1

            # break up the sql commands to 90 rows each
            if thresholdVal == 90 or i == len(prediction):
                predictUploadQuery = predictUploadQuery[:-1] #remove last comma
                predictUploadQuery += ";"                      
                cursor.execute(predictUploadQuery)
                db_connection.commit()
                thresholdVal = 0
                predictUploadQuery = "INSERT INTO predictions VALUES "             
            thresholdVal += 1      
    cursor.close()
    db_connection.close()

    return batchName