# Basic modules for dataframe manipulation
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import iqr
from pandas.api.types import is_string_dtype, is_numeric_dtype, is_categorical_dtype
import math


def lower_cols(dataframe):
	"""
	Function used to convert column headings to lower case
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	
	"""
	dataframe.columns = [x.lower() for x in dataframe.columns]


def obj_to_cat(dataframe):
   """
	Function used to convert objects(strings) into categories
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	
	"""

   for n, c in dataframe.items():
        if is_string_dtype(c):
            dataframe[n] = c.astype('category').cat.as_ordered()
   return dataframe
	


def fill_missing_nums(dataframe):    
    """
	 Function used to impute missing numerical values with column's median
	
	 Parameters:
	
	 dataframe - just as the parameter name implies, expects dataframe object
	
   	 """
    
    for n, c in dataframe.items(): 
        if is_numeric_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.median())
    return dataframe


def fill_missing_cats(dataframe):
    """
    Function used to impute missing categorical values with column's mode
	
    Parameters:
	
    dataframe - just as the parameter name implies, expects dataframe object
	
    """
    for n, c in dataframe.items():
        if is_categorical_dtype(c):
            if pd.isnull(c).sum() > 0:
                dataframe.loc[:,n] = c.fillna(c.mode()[0])
    return dataframe
	


def display_cols(dataframe, type = 'category', num_samples = 7):
	"""
	Function used to display columns of desired data type
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	type - data type we are looking for
	num_samples - number of rows to display
	
	"""
	mask = dataframe.dtypes == type
	return dataframe.loc[:, mask].sample(num_samples)
	


def display_nums_stats(dataframe):
    """
    Function used to calculate basic statistics of numerical columns.
	
    Parameters:
	
    dataframe - just as the parameter name implies, expects dataframe object
	
    """

    numericals = []
    for n, c in dataframe.items():
	    if is_numeric_dtype(c):
		    numericals.append(n)
    return dataframe[numericals].describe()


	
def outliers_by_col(dataframe, train_last_idx , multiplier = 1.5, plot_results = True, outliers_dictionary = False):
	"""
	Function used to determine outliers in each column.
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	multiplier - value used for calculating Tukey's Interquartile Range. By default we assume that all values lower than Q1 - (1.5 * IQR)  or greather than Q3 + (1.5 * IQR) are outliers
	plot_results - by default set to True. As a result boxplots for all columns with outliers will be plotted
	outliers_dictionary - by default set to False. If True, dictionary with column names as keys and lists of row indexes containing outliers as values will be returned 
	
	"""
	
	outliers_dict = {}
	for column in dataframe.columns:
			if is_numeric_dtype(dataframe[column][:train_last_idx]):
				iq_range = iqr(dataframe[column][:train_last_idx])
				q1 = np.percentile(dataframe[column][:train_last_idx], 25)
				q3 = np.percentile(dataframe[column][:train_last_idx], 75)
				lower_bound = q1 - (multiplier * iq_range)
				upper_bound = q3 + (multiplier * iq_range)
				select_indices = list(np.where((dataframe[column][:train_last_idx] < lower_bound) | (dataframe[column][:train_last_idx] > upper_bound))[0])
				if len(select_indices) > 0 :
					outliers_dict[column] = select_indices

	
	if plot_results == True:
		plot_categoricals(dataframe[:train_last_idx], outliers_dict.keys(), kind = 'box', figsize = (20,10))
		
	if outliers_dictionary == True:
		return outliers_dict



	
def nominalnums_to_cat(dataframe, unique_values_split = 30,  boundary = 10):
	"""
	Function for converting nominal numerical features into categorical variables. 
	
	Parameters:
	
	dataframe -just as the parameter name implies, expects dataframe object
	unique_values_split - number of unique values to treat a variable as a categorical one. By default, variable's data type will be changed to 'category' if it has less than 30 unique values
	boundary - decision boundary determining which variable names will be returned in list for further check. By default, all variables which take more than 10 unique values will be returned
	
	"""
	cols_to_verify = []
	for col in dataframe.columns:
		if is_numeric_dtype(dataframe[col]):
			length = len(dataframe[col].value_counts())
			if ((length < unique_values_split) and ('area' not in col)):
				dataframe[col] = dataframe[col].astype('category')
				if (length > boundary):
					cols_to_verify.append(col)
	return(cols_to_verify)
	

def plot_categoricals(dataframe, columns, kind = 'count', figsize = (20,10)):
	"""
	Function for plotting suspicious categorical columns

	Parameters:

	dataframe - just as the parameter name implies, expects dataframe object
	columns - list of columns or dictionary keys, e.g. list of columns returned by 'nominalnums_to_cat' function
	kind - by default set to 'count' to display countplots for given columns. If 'box' will be used as a value then function will display box plots. 

	"""
	
	length = len(columns)
	if length <= 6:
		plt.figure(figsize=figsize)
	elif length > 6 and length <= 12:
		plt.figure(figsize = next((x, int(y*2)) for x,y in [figsize]))
	elif length > 12 and length < 18:
		plt.figure(figsize = next((x, int(y*3)) for x,y in [figsize]))
	elif length > 18 and length < 24:
		plt.figure(figsize = next((x, int(y*3)) for x,y in [figsize]))
	for ix, col in enumerate(columns):
		plt.subplot(np.ceil(length/3), 3, ix+1)
		if kind == 'count':
			sns.countplot(dataframe[col])
		elif kind == 'box':
			sns.boxplot(dataframe[col])
			
			
def binarize_numericals(dataframe, columns):
	"""
	Function for creating binomial categorical variables from unequally distributed numerical variables, all values equal to 0 will be denoted as 0 and those greater than 0 will be marked as 1. After conversion, all input variables will be dropped from dataframe.

	Parameters:

	dataframe - just as the parameter name implies, expects dataframe object
	columns - list of columns or dictionary keys to convert

	"""
	for col in columns:
		dataframe[col+'_bin'] = np.where(dataframe[col] > 0, 1, 0)
		dataframe[col+'_bin'] = dataframe[col+'_bin'].astype('category')

	dataframe.drop(labels= columns, axis=1, inplace = True)

		
		
def get_codes(dataframe):
	"""
	Function for converting values of categorical variables into numbers
	
	Parameters:
	
	dataframe - just as the parameter name implies, expects dataframe object
	
	"""
	for column in dataframe.columns:
		if is_categorical_dtype(dataframe[column]):
			dataframe[column] = dataframe[column].cat.codes
			

def rmsle(real, predicted):
    sum=0.0
    for x in range(len(predicted)):
        if predicted[x]<0 or real[x]<0: #check for negative values
            continue
        p = np.log(predicted[x]+1)
        r = np.log(real[x]+1)
        sum = sum + (p - r)**2
    return (sum/len(predicted))**0.5
		
def rmsle2(y, y0):
    assert len(y) == len(y0)
    return np.sqrt(np.mean(np.power(np.log1p(y)-np.log1p(y0), 2)))
	
	
def rmse(x,y): return math.sqrt(((x-y)**2).mean())


def print_score(model, X_train, X_val, y_train, y_val):
	"""
	Function used for checking the accuracy of the regression model
	
	Parameters:
	
	model -just as the parameter name implies, expects model object
	X_train - training subset of explanatory variables
	X_val - validation subset of explanatory variables
	y_train - training subset of target variable
	y_val - validation subset of target variable
	
	"""
	res = [rmse(model.predict(X_train), y_train), rmse(model.predict(X_val), y_val), model.score(X_train, y_train), model.score(X_val, y_val)]
	print(res)
	
	
	
def plot_feat_imp(model, dataframe):

	"""
	Function used for plotting the most important features found by model
	
	Parameters:
	
	model - just as the parameter name implies, expects model object
	dataframe - just as the parameter name implies, expects dataframe object

	
	"""
	indices = np.argsort(model.feature_importances_)[::-1][:15]

	fig = plt.figure(figsize=(6, 9))
	p = sns.barplot(y=dataframe.columns[indices][:15], x = model.feature_importances_[indices][:15], orient='h')
	p.set_xlabel("Relative importance",fontsize=12)
	p.set_ylabel("Features",fontsize=12)
	p.tick_params(labelsize=9)
	p.set_title("Selected classifier feature importance")

	plt.show()
	
