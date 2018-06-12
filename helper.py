# Basic modules for dataframe manipulation
import numpy as np
import pandas as pd
from scipy.stats import iqr
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype



# Converting objects(strings) into categories
def obj_to_cat(dataframe):
    for n, c in dataframe.items():
	    if is_string_dtype(c):
		    dataframe[n] = c.astype('category').cat.as_ordered()
    return dataframe
	
	
def fill_missing_nums(dataframe):    
    for n, c in dataframe.items(): 
        if is_numeric_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.median())
    return dataframe
	
def fill_missing_cats(dataframe):
    for n, c in dataframe.items():
        if is_categorical_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.mode()[0])
    return dataframe
	

def display_cols(dataframe, type = 'category', num_samples = 7):
	mask = dataframe.dtypes == type
	return dataframe.loc[:, mask].sample(num_samples)
	

def display_nums_stats(dataframe):
    numericals = []
    for n, c in dataframe.items():
	    if is_numeric_dtype(c):
		    numericals.append(n)
    return dataframe[numericals].describe()

	
def outliers_by_col(dataframe, multiplier = 1.5):
	outliers_dict = {}
	for column in dataframe.columns:
			if is_numeric_dtype(dataframe[column]):
				iq_range = iqr(dataframe.loc[:, column])
				q1 = np.percentile(dataframe.loc[:, column], 25)
				q3 = np.percentile(dataframe.loc[:, column], 75)
				lower_bound = q1 - (multiplier * iq_range)
				upper_bound = q3 + (multiplier * iq_range)
				select_indices = list(np.where((dataframe[column] < lower_bound) | (dataframe[column] > upper_bound))[0])
				if len(select_indices) > 0 :
					outliers_dict[column] = select_indices
	return outliers_dict


	
def nominalnums_to_cat(dataframe, columns):
    for column in columns:
        dataframe[column] = dataframe[column].astype('category')
    return dataframe