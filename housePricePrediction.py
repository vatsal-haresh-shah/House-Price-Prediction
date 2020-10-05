import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('train.csv')

df.columns

# Analysing SalePrice
df['SalePrice'].describe()
sns.distplot(df['SalePrice'])
print("Skewness : %f" % df['SalePrice'].skew())
print("Kurtosis: %f" % df['SalePrice'].kurt())

#Analysing SalePrice with numerical variable
var = 'GrLivArea'
data = pd.concat([df['SalePrice'],df[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim=(0,800000));
#There is a linear relation 

var = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'],df[var]], axis = 1)
data.plot.scatter(x = var, y = 'SalePrice', ylim=(0,800000));
#There is an exponential relation

#Analysing SalePrice with Categorical Variable
var = 'OverallQual'
data = pd.concat([df['SalePrice'],df[var]], axis= 1)
f, ax = plt.subplots(figsize=(8,6))
fig = sns.boxplot(x = var, y = 'SalePrice', data = data)
fig.axis(ymin = 0, ymax = 800000);

var = 'YearBuilt'
data = pd.concat([df['SalePrice'],df[var]], axis= 1)
f, ax = plt.subplots(figsize=(16, 8))
fig = sns.boxplot(x = var, y = 'SalePrice', data = data)
fig.axis(ymin = 0, ymax = 800000);
plt.xticks(rotation = 90)

#Correlation Matrix
corrmat = df.corr()
f, ax = plt.subplots(figsize = (12, 9))
sns.heatmap(corrmat, vmax =.8, square = True);

#SalePrice Correlation Matrix
k = 10
cols = corrmat.nlargest(k, 'SalePrice')['SalePrice'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.25)
hm = sns.heatmap(cm, cbar=True, annot=True, square=True, fmt='.2f', annot_kws={'size' : 10},
				 yticklabels=cols.values, xticklabels=cols.values)
plt.show()

#Scatter Plot
sns.set()
cols = ['SalePrice','OverallQual','GrLivArea','GarageCars','TotalBsmtSF','FullBath','YearBuilt']
sns.pairplot(df[cols], size = 2.5)
plt.show();


#Missing Values
total = df.isnull().sum().sort_values(ascending=False)
percent = (df.isnull().sum()/df.isnull().count()).sort_values(ascending=False)
missing_data = pd.concat([total,percent], axis = 1, keys = ['Total','Percent'])
missing_data.head(20)

#Dealing with Missing Data
df = df.drop((missing_data[missing_data['Total'] > 1]).index,1)
df = df.drop(df.loc[df['Electrical'].isnull()].index)
df.isnull().sum().max()

#Standardizing Data
SalePrice_scaled = StandardScaler().fit_transform(df['SalePrice'][:,np.newaxis]);
low_range = SalePrice_scaled[SalePrice_scaled[:,0].argsort()][:10]
high_range = SalePrice_scaled[SalePrice_scaled[:,0].argsort()][-10:]
print('outer range (low) of the distribution')
print(low_range)
print('\n Outer range (High) of the distribution')
print(high_range)

#Bivarite Analysis SalePrice/GrLivArea
var = 'GrLivArea'
data = pd.concat([df['SalePrice'],df[var]], axis=1)
data.plot.scatter(x = var, y = 'SalePrice', ylim=(0,800000));

#deleting outliars from GrLivArea
df.sort_values(by = 'GrLivArea', ascending = False )[:2]
df = df.drop(df[df['Id'] == 1299].index)
df = df.drop(df[df['Id'] == 524].index)

#Bivariate analysis of SalePrice/TotalBsmtSF
var = 'TotalBsmtSF'
data = pd.concat([df['SalePrice'],df[var]], axis = 1)
data.plot.scatter(x=var, y = 'SalePrice', ylim = (0,800000));

#Histograme and normal probability plot for SalePrice
sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)

#Applying Log transformation
df['SalePrice'] = np.log(df['SalePrice'])

#Transformed Histograme and normal probability plot for SalePrice
sns.distplot(df['SalePrice'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['SalePrice'], plot=plt)

#Histograme and normal probability plot of GrLivArea
sns.distplot(df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)

#Applying Log transformation
df['GrLivArea'] = np.log(df['GrLivArea'])

#Transformed Histograme and normal probability plot 
sns.distplot(df['GrLivArea'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['GrLivArea'], plot=plt)

#Histograme and normal probability plot of TotalBsmtSF
sns.distplot(df['TotalBsmtSF'], fit=norm);
fig = plt.figure()
res = stats.probplot(df['TotalBsmtSF'], plot=plt)

#Creating an extra column Named HasBsmt
df['HasBsmt'] = pd.Series(len(df['TotalBsmtSF']), index = df.index)
df['HasBsmt'] = 0
df.loc[df['TotalBsmtSF']>0, 'HasBsmt'] = 1

#log transformation of TotalBsmtSF
df.loc[df['HasBsmt'] == 1 , 'TotalBsmtSF'] = np.log(df['TotalBsmtSF'])

#Histogram and normal Probability plot for trnasformed TotalBsmtSF
sns.distplot(df[df['TotalBsmtSF']>0]['TotalBsmtSF'], fit = norm)
fig = plt.figure()
res = stats.probplot(df[df['TotalBsmtSF']>0]['TotalBsmtSF'], plot = plt)

#scatter plot
plt.scatter(df['GrLivArea'],df['SalePrice']);

#scatter plot
plt.scatter(df[df['TotalBsmtSF']>0]['TotalBsmtSF'], df[df['TotalBsmtSF']>0]['SalePrice']);

#convert categorical variable into dummy
df = pd.get_dummies(df)

df.to_csv(r'train_transformed.csv', index=False, header=True)


dfTest = pd.read_csv('test.csv')

testTotal = dfTest.isnull().sum().sort_values(ascending=False)
percentTotal = (dfTest.isnull().sum()/dfTest.isnull().count()).sort_values(ascending=False)
testMissing = pd.concat([testTotal,percentTotal], axis=1, keys=['Total','Percent'])
testMissing.head(20)

dfTest.columns

missing_data.head(20)
