# -*- coding: utf-8 -*-
"""
Created on Tue Sep 15 09:09:10 2020

@author: Vatsal Shah
"""
# Importing necessary libraries 
import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# Importing the data set from my home directory 
train  = pd.read_csv('train.csv')
test = pd.read_csv('test.csv')
# Dimensions of test and train dataset
train.shape
test.shape

# columns in test and train
train.columns
test.columns

#data cleaning and dealing with missing values.

train.isnull().sum().sort_values(ascending = False)
test.isnull().sum().sort_values(ascending = False)
#Precentage of missing values in training dataset
total_train = train.isnull().sum().sort_values(ascending = False)
percent_train = (train.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_train = pd.concat([total_train,percent_train], axis = 1, keys = ['Total','Percent'])
missing_train.head(20)
#Precentage of missing values in testing dataset
total_test = test.isnull().sum().sort_values(ascending = False)
percent_test = (test.isnull().sum()/train.isnull().count()).sort_values(ascending = False)
missing_test = pd.concat([total_test, percent_test], axis  = 1, keys = ['Total','Precentage'])
missing_test.head(20)

#Dealing with missing values
#PoolQC

def to_numerical(x):
	if x == 'None':
		return 0
	if x == 'Po':
		return 1
	if x == 'Fa':
		return 2
	if x == 'TA':
		return 3
	if x == 'Gd':
		return 4
	if x == 'Ex':
		return 5

train['PoolQC'].unique()
train['PoolQC'].fillna('None', inplace = True)
train['PoolQC'] = train['PoolQC'].apply(to_numerical)

test['PoolQC'].fillna('None', inplace  = True)
test['PoolQC'] = test['PoolQC'].apply(to_numerical)
train['PoolQC'].isna().sum()
#MiscFeature
train['MiscFeature'].unique()
train['MiscFeature'].fillna('None', inplace = True)
train['MiscFeature'] = train['MiscFeature'].astype('category')
test['MiscFeature'].fillna('None', inplace = True)
test['MiscFeature'] = test['MiscFeature'].astype('category')
train['MiscFeature'].isna().sum()
#Alley
train['Alley'].dtypes
train['Alley'].unique()
train['Alley'].fillna('None', inplace = True)
train['Alley'] = train['Alley'].astype('category')
test['Alley'].fillna('None', inplace = True)
test['Alley'] = test['Alley'].astype('category')
train['Alley'].isna().sum()
#Fence
train['Fence'].unique()
train['Fence'].fillna('None', inplace = True)
train['Fence'] = train['Fence'].astype('category')
test['Fence'].fillna('None', inplace  = True)
test['Fence'] = test['Fence'].astype('category')

#FireplaceQu
train['FireplaceQu'].unique()
train['FireplaceQu'].fillna('None', inplace = True)
train['FireplaceQu'] = train['FireplaceQu'].apply(to_numerical)

test['FireplaceQu'].fillna('None', inplace = True)
test['FireplaceQu'] = test['FireplaceQu'].apply(to_numerical)
train['FireplaceQu'].isna().sum()

# Garage
g = [c for c in train if 'Garage' in c]
train[g].isna().sum()
test[g].isna().sum()
train[g].dtypes
for col in ['GarageType','GarageFinish','GarageQual', 'GarageCond']:
	train[col].fillna('None', inplace = True)
	test[col].fillna('None', inplace = True)
train[g].isna().sum()
test[g].isna().sum()
train['GarageYrBlt'].fillna(0, inplace = True)
test['GarageYrBlt'].fillna(0, inplace = True)

for col in ['GarageQual', 'GarageCond']:
	train[col] = train[col].apply(to_numerical)
	test[col] = test[col].apply(to_numerical)

train[g].isna().sum()
train['GarageType'] = train['GarageType'].astype('category')
train['GarageFinish'] = train['GarageFinish'].astype('category')
test['GarageType'] = test['GarageType'].astype('category')
test['GarageFinish'] = test['GarageFinish'].astype('category')

#Basement
b = [c for c in train if 'Bsmt' in c]
train[b].isna().sum()
test[b].isna().sum()
for col in ['BsmtQual','BsmtCond','BsmtExposure','BsmtFinType1','BsmtFinType2']:
	train[col].fillna('None', inplace = True)
	test[col].fillna('None', inplace = True)
test[b].isna().sum()
train[b].isna().sum()

test['BsmtFullBath'].isna().sort_values(ascending = False)
test = test.drop(660, axis = 0)
test = test.drop(728, axis = 0)

for col in ['BsmtQual', 'BsmtCond']:
	train[col] = train[col].apply(to_numerical)
	test[col] = test[col].apply(to_numerical)

#MasVnrType replacing with mode value
train['MasVnrType'].fillna(train['MasVnrType'].mode(), inplace = True)
test['MasVnrType'].fillna(test['MasVnrType'].mode(), inplace = True)

# Deleting the rest 
train.isna().sum().sort_values(ascending = False).head(10)
test.isna().sum().sort_values(ascending = False).head(20)
train = train.dropna()
test = test.dropna()

train_num = train.select_dtypes(include = ['int16','int32','int64','float16','float32','float64'])
train_c = train.select_dtypes(include = ['category','object'])
corr = train_num.corr()
plt.subplots(figsize = (15,12))
sns.heatmap(corr, vmax = 0.9, cmap = "Blues", square = True)