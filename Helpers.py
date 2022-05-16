# Helper Functions

import pandas as pd
import math
import itertools
import operator
import matplotlib.pyplot as plt

def windowDf(df, shift):
    width = 20
    new_df = pd.DataFrame([], columns = ['Linear X', 'Linear Y', 'Linear Z', 'Linear Resultant', 'Rotational X', 'Rotational Y', 'Rotational Z', 'Rotational Resultant'])
    # take the merged linear and rotational data and find the range around the max for each column
    for col in df.columns:        
        if abs(df[col].max()) > abs(df[col].min()):
            extremeRow = df[col].idxmax()
        else:
            extremeRow = df[col].idxmin()        
        
        extremeRow = extremeRow + shift
        
        if extremeRow - width < 0:                    
            df_temp = pd.DataFrame([-123]*(width-extremeRow))
            df_temp2 = df.loc[0:width+extremeRow, [col]]
            df_temp.columns = df_temp2.columns

            df_temp = pd.concat([df_temp, df_temp2], ignore_index=True) 
        
            new_df_avg = df[col].mean()
            df_temp.loc[0:width-extremeRow-1, col] = new_df_avg  

            new_df[col] = df_temp[col].values                      

        elif extremeRow + width >= 120:            
            df_temp = pd.DataFrame([-123]*(width+extremeRow-120+1))                     
            df_temp2 = df.loc[extremeRow-width:120-1, [col]]
            df_temp.columns = df_temp2.columns
            
            df_temp = pd.concat([df_temp2, df_temp], ignore_index=True)            
            
            new_df_avg = df[col].mean()            
            df_temp.loc[120-extremeRow+width:width*2, col] = new_df_avg              
        
            new_df[col] = df_temp[col].values                                    

        else:                       
            new_df[col] = df[col].iloc[extremeRow-width:extremeRow+width+1].values
     
    return new_df


def resolve_NAs(df, colStr):
    # change all NA values in specified column to be value of the previous row
    for index, row in df.iterrows():
        val = df.at[index, colStr]
        
        if index > 0:
            if math.isnan(val):                        
                df.at[index, colStr] = df.at[index-1, colStr]        
                
    return df


def dbSensorData_to_dfs(sensorData, isLinear):
    sensorDataDfs = []

    # sensorData contains all caseIDs, so need to seperate out into different lists for each caseID
    # sort by impactID
    for mid, group in itertools.groupby(sorted(sensorData,key=operator.itemgetter(1)), key=operator.itemgetter(1)): 
        if isLinear:
            df = pd.DataFrame(list(group), columns =['frameID', 'caseID', 'Impact Date', 'Impact Time', 'Linear X', 'Linear Y', 'Linear Z', 'Linear Resultant'], index=None)                        
        else:
            df = pd.DataFrame(list(group), columns =['frameID', 'caseID', 'Impact Date', 'Impact Time', 'Rotational X', 'Rotational Y', 'Rotational Z', 'Rotational Resultant'], index=None)
            
        # remove unncessary columns
        df.drop(['frameID'], axis = 1, inplace=True)
        
        sensorDataDfs.append(df)                
    
    return sensorDataDfs
    
def femData_to_df(femData):    
    
    ###### combine individual output types into one tuple for entire caseID ######
    # empty dict
    result = {}
    # iterating over the tuples
    for sub_tuple in femData:
        # checking the first element of the tuple in the result
        if sub_tuple[0] in result:
            # adding the current tuple values without first one
            result[sub_tuple[0]] = (*result[sub_tuple[0]], *sub_tuple[1:])
        else:
            # adding the tuple
            result[sub_tuple[0]] = sub_tuple
    
    # remove type names from the tuples
    result = [(case, CorpCallval, Thalval, brainVal) for case, name, brainVal, name2, CorpCallval, name3, Thalval in list(result.values())]    

    df = pd.DataFrame(result, columns =['Case', "Corpus Callosum", "Thalamus", "Brain Stem"], index=None)
    return df

def merge_linear_rotational(df_linearData, df_rotationData):   
    # remove unncessary column
    df_linearData.drop(['caseID'], axis = 1, inplace=True) 
    df_rotationData.drop(['caseID'], axis = 1, inplace=True) 

    # use full outer merge to include all linear and rotational data points -  even if they aren't matched to each other
    df_combined = pd.merge(df_linearData, df_rotationData, on='Impact Time', how='left')
    
    # remove unncessary columns
    df_combined.drop(['Impact Date_x', 'Impact Date_y', 'Impact Time'], axis = 1, inplace=True)
    
    featureColStrList = ['Linear X', 'Linear Y', 'Linear Z', 'Linear Resultant', 'Rotational X', 'Rotational Y', 'Rotational Z', 'Rotational Resultant']
    
    for featureStr in featureColStrList:
        df_combined = resolve_NAs(df_combined, featureStr)
        
    # release allocated memory
    del df_linearData
    del df_rotationData
    
    return df_combined


# Function for when using filePaths (local excel files)
def merge_linear_rotational2(filePath):
    # sheet_name 0 refers to the linear data
    df_linearData = pd.read_excel(filePath, sheet_name=0)
    
    # sheet_name 1 refers to the rotational data
    df_rotationData = pd.read_excel(filePath, sheet_name=1)
    
    # use full outer merge to include all linear and rotational data points -  even if they aren't matched to each other
    df_combined = pd.merge(df_linearData, df_rotationData, on='Impact Time', how='left')
    
    # remove unncessary columns
    df_combined.drop(['Impact Date_x', 'Impact Date_y', 'Impact Time'], axis = 1, inplace=True)
    
    featureColStrList = ['Linear X', 'Linear Y', 'Linear Z', 'Linear Resultant', 'Rotational X', 'Rotational Y', 'Rotational Z', 'Rotational Resultant']
    
    for featureStr in featureColStrList:
        df_combined = resolve_NAs(df_combined, featureStr)
        
    # release allocated memory
    del df_linearData
    del df_rotationData
    
    return df_combined

def normalize_min_max(inputs_df):
    normalized_inputs = (inputs_df - inputs_df.min()) / (inputs_df.max() - inputs_df.min())
    return normalized_inputs

def reverse_normalize(normalized_df, origMax, origMin):
    original_df = normalized_df*(origMax - origMin) + origMin
    return original_df

def visualize_loss(history, title):
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    epochs = range(len(loss))
    plt.figure()
    plt.plot(epochs, loss, "b", label="Training loss")
    plt.plot(epochs, val_loss, "r", label="Validation loss")
    plt.title(title)
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.show()