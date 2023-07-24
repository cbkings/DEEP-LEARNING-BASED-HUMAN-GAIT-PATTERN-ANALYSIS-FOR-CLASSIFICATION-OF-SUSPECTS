import numpy as np
import pandas as pd
import pprint, pickle
#from pycaret.utils import version
#version()
import pandas as pd
import numpy as np
#from itertools import chain
from sklearn.ensemble import IsolationForest
from sklearn.impute import KNNImputer

import os

# Define the file paths

def feature_gen(cam_angle):
    s_file_S_path = 'valid_para/Static_{}.pkl'.format(cam_angle)
    s_file_D_path = 'valid_para/Dynamic_{}.pkl'.format(cam_angle)
    # Load the files if they exist
    s_file_S = open(s_file_S_path, 'rb')
    s_file_D = open(s_file_D_path, 'rb')
    sT_S = pickle.load(s_file_S)
    sT_D = pickle.load(s_file_D)
    s_file_S.close()
    s_file_D.close()

    merged_df=[]
    gait_merged=[]
    data=[]
    data_missing=[]
    imputer=[]
    data_imputed=[]
    data_NaN_handled=[]
    model=[]
    outliers=[]
    data_outliered=[]
    time=[]
    output=[]

    # merge the two dataframes based on the common 'time' column
    merged_df = pd.merge(sT_S, sT_D, on='time', how='outer')

    # print the merged dataframe
    merged_df = merged_df.sort_values(by='time')
    merged_df = merged_df.reset_index(drop=True)
    count_col = merged_df.pop("count")
    merged_df.insert(0, "count", count_col)

    merged_df

    ## drop count & time columns
    gait_merged=merged_df.iloc[:, 2:]
    gait_merged

    #handle missing NAN values
    data = gait_merged


    # Loop over each column in the dataset
    for col in data.columns:
        # Calculate mean of non-NaN values in the current column
        col_mean = data[col][~data[col].isna()].mean()

        # Replace NaN values in the current column with the calculated mean
        data[col] = data[col].fillna(col_mean)

    # Apply mean imputation to handle missing values

    # Convert the imputed data back to a Pandas DataFrame
    data_NaN_handled = pd.DataFrame(data, columns=data.columns)

    # Save the imputed data to a CSV file
    data_NaN_handled

    #Handling Outliers

    # Define the Isolation Forest model with a contamination parameter of 0.05 (i.e., 5% of data points are expected to be outliers)
    model = IsolationForest(n_estimators=100, max_samples='auto', contamination=0.05, random_state=42)

    # Fit the model to the data and predict the outliers
    model.fit(data_NaN_handled)
    outliers = model.predict(data_NaN_handled)

    # Remove the outliers from the dataset
    data_outliered = data_NaN_handled[outliers == 1]


    time = merged_df[['time']]

    # merge time dataframe with data_outliered dataframe
    data_outliered.insert(0, 'time', merged_df['time'])
    data_outliered = data_outliered.reset_index(drop=True)


    # show data_outliered dataframe with the corresponding time column

    output = open('valid_feature_vector/FV_evidance_{}.pkl'.format(cam_angle), 'wb')
    pickle.dump(data_outliered, output)
    output.close()

