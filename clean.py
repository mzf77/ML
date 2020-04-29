import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas import Series,DataFrame

df=pd.read_csv('/home/anemias/Downloads/1950-2018_all_tornadoes.csv', index_col='date', parse_dates=True)
 #dtype={'mag': np.object,'fc':np.object}
df.head(2)
df.columns
#drop columns
df.drop(columns=['om', 'mo','dy','tz','stf','stn', 'slat','slon','elat','elon','ns','sg','sn','fc','closs'],axis = 1, inplace=True)

#year
df['year']=df[{'yr':'year'}]

#st 
df.dtypes
df.st.value_counts() >100
# df['st'] > 100
# df.st.head()
#data.desc.head().apply(lambda x: x.lower())
# df.st.head().apply(lambda x: x.upper())

#mag
df.hist('mag')
df_outliers = df[(df.mag > df.mag.quantile(.005))]
df_outliers.mag.replace(-9,np.nan, inplace=True)
df_outliers.mag.value_counts()
#df.mag.replace(0,np.nan, inplace=True)
df_outliers.isnull().sum()
#df.dropna(inplace=True, axis=0)
df_outliers.shape
df_outliers.mag.plot.hist()
df_outliers.boxplot('mag')

# df.mag.replace(1,str('f1'), inplace=True)
# df.mag.replace(2,str('f2'), inplace=True)
# df.mag.replace(3,str('f3'), inplace=True)
# df.mag.replace(4,str('f4'), inplace=True)
# df.mag.replace(5,str('f5'), inplace=True)

df_majormag=df_outliers[(df_outliers['mag'])>=2]
df_majormag.mag.plot.hist()
df_majormag.mag.describe(include='object')
df_majormag.boxplot('mag')
df_majormag.shape
#change data type 
df_majormag.mag = pd.to_numeric(df_majormag.mag, errors = 'coerce')

#time 

#inj
df_majormag.inj.describe(include='all')
df_majormag.inj.plot(kind='line')
# df_injuries=df_majormag[df_majormag['inj']>0]
#df_injuries.inj.plot.line()

#fat
df_majormag.fat.describe()
df_majormag.fat.plot()
df_majormag.fat>0

#check NULL values
df_majormag.isnull().sum()
df_majormag.isnull().any()
df_majormag.isnull().sum()/ df.shape[0]
df_majormag.shape
#print
print('Num of NUN values',df_majormag.isnull().sum())
#Drop Null values if exists 
#df.dropna(axis=0, inplace=True)

#
df_majormag['st'].astype('str')
df_majormag.dtypes

#remove columns with certain threshold of nulls
#threshold is the number of columns or rows without nulls 
# df.len
# thresh = len(df)*.6
# df.dropna(thresh = thresh, axis = 1)
# df.dropna(thresh = 21, axis = 0)

# df.mag.dtype
# df.mag.value_counts()

# df.mag = df.mag.apply(lambda x: str(x).replace('mag','').strip())
# df.mag.value_counts()

#bar chart of types 
df_majormag.dtypes.value_counts().plot(kind='bar')

#view columns & rename columns 
df_majormag.columns
df_majormag.head(2)
# resetting index 
df_majormag.reset_index(inplace = True) 

#remove duplcates, subset, keep, etc.
df_majormag.duplicated().sum()
df_majormag.drop_duplicates()

df_majormag.columns

#filter for multiple columns (all below do the same thing ) 
df_filter=df_majormag[['date','year','mag','st','loss','inj','fat','len']]
df_filter.loc[:,['date','mag','len']]
df_filter.iloc[:,0:3]

#drop / add column #inplace = True 
#axis & inplace 
#df.drop('inj', axis = 1)
#data.drop(['url','price'], axis = 1)
#plotting

for i in df_filter.columns:
    cat_num = df_filter[i].value_counts()
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()


for i in df_filter[['mag','inj','fat']].columns:
    cat_num = df_filter[i].value_counts()[:20]
    print("graph for %s: total = %d" % (i, len(cat_num)))
    chart = sns.barplot(x=cat_num.index, y=cat_num)
    chart.set_xticklabels(chart.get_xticklabels(), rotation=90)
    plt.show()

#tenary operator 
#Filter Damage 
df_filter['DM']=df_filter['mag'].apply(lambda x: 'MD' if x > 4 else 'LD')
df_filter.DM.unique()
df_filter.DM.describe(include='object')
df_filter.head(2)
#Filter Fatalities
df_filter['Fatalities'] = df_filter[['inj','fat']].apply(lambda x: 'yes' if x[0] > 100 and x[1] < 1000 else 'no', axis = 1)
df_filter.Fatalities.count

#Change date format
df_filter['date']=pd.to_datetime(df_filter['date'])
df_filter.date.dtype
df_filter.head()
#dummy variables
 
df_dummies = pd.get_dummies(df_filter[['mag','st','inj','fat','loss','DM','Fatalities']])
df_dummies

#pivot table / sort_index / sort_values 
df_majormag.pivot_table(index='date',columns='mag',values='inj',aggfunc ='mean').sort_index(ascending=False)

df_majormag.pivot_table(index='date',columns='st',values='mag',aggfunc ='count').sort_index(ascending=False)
pd.pivot_table(df_filter, index = ['date','st'], values = 'inj')
pd.pivot_table(df_filter, index = ['date','st'], values = 'inj').sort_values('inj', ascending = False)
df_majormag.pivot_table(index='date',columns='mag',values='fat',aggfunc ='count').sort_index(ascending=False).plot()



#groupby 
df_filter.columns
df_filter.groupby('mag').mean().head(9)
df_filter.groupby('DM').mean()
df_filter.groupby('year').mean()

df_filter.groupby(['mag','inj']).mean()
df_filter.groupby(['year','inj']).mean()
df_filter.groupby(['fat','inj']).mean()
df_filter.groupby(['len','loss']).mean()
#Not index
df_filter.groupby(['mag','inj'],as_index = False).mean()

#binning pd.cut / pd.qcut
#pd.qcut(df_outliers.mag,5.0) #even number 
#pd.cut(data.price,5 ) #even spacing 

# with no null values 
from scipy import stats
import numpy as np 

#types of normalization 
from sklearn.preprocessing import MinMaxScaler
scaler = MinMaxScaler()
scaler.fit(df_filter.mag.values.reshape(-1,1))
scaler.transform(df_filter.inj.values.reshape(-1,1))

# write to a csv file pd.to_csv()
df_filter.to_csv('Clean_data_torna.csv')
clean=pd.read_csv('Clean_data_torna.csv')
clean.head(5)
#Exploration and model
numeric = df_filter._get_numeric_data()

import seaborn as sns

corrdata = numeric
corr = corrdata.corr()
ax = sns.heatmap(
    corr, 
    vmin=-1, vmax=1, center=0,
    cmap=sns.diverging_palette(20, 220, n=200),
    square=True
)
ax.set_xticklabels(
    ax.get_xticklabels(),
    rotation=45,
    horizontalalignment='right'
);

#simple linear regression for year
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import math

#df_filter.mag.unique()
#df_filter= pd.Series.add_prefix(self= ~df_filter.mag,prefix='F_')

#set variables need to be in specific format 
X1 = df_filter.fat.values.reshape(-1,1)
y1 = df_filter.inj.values.reshape(-1,1)

#create train / test split for validation 
X_train1, X_test1, y_train1, y_test1 = train_test_split(X1, y1, test_size=0.3, random_state=0)
        
reg = LinearRegression().fit(X_train1, y_train1)
reg.score(X_train1, y_train1)
reg.coef_
y_hat1 = reg.predict(X_train1)

plt.scatter(X_train1,y_train1)
plt.scatter(X_train1,y_hat1)
plt.show()

y_hat_test1 = reg.predict(X_test1)
plt.scatter(X_test1, y_test1)
plt.scatter(X_test1, y_hat_test1)
plt.show()

#MSE & RMSE penalize large errors more than MAE 
mae = mean_absolute_error(y_hat_test1,y_test1)
rmse = math.sqrt(mean_squared_error(y_hat_test1,y_test1))
print('Root Mean Squared Error = ',rmse)
print('Mean Absolute Error = ',mae)

import statsmodels.api as sm

X1b = df_filter[['mag','fat']]
y1b = df_filter.inj.values

X_train1b, X_test1b, y_train1b, y_test1b = train_test_split(X1b, y1b, test_size=0.3, random_state=0)

reg_sm1b = sm.OLS(y_train1b, X_train1b).fit()
reg_sm1b.summary()

df_filter.dtypes
#multiple linear regression 
from statsmodels.stats.outliers_influence import variance_inflation_factor

X2 = df_filter[['year','mag','fat','loss','len']]
y2 = df_filter.inj.values

X_train2, X_test2, y_train2, y_test2 = train_test_split(X2, y2, test_size=0.3, random_state=0)

reg_sm2 = sm.OLS(y_train2, X_train2).fit()
reg_sm2.summary()

pd.Series([variance_inflation_factor(X2.values,i) for i in range(X2.shape[1])],index=X2.columns)

#actual regression 
X3 = pd.get_dummies(df_filter[['mag','fat','loss','len','st','year']])
y3 = df_filter.inj.values

X_train3, X_test3, y_train3, y_test3 = train_test_split(X3, y3, test_size=0.3, random_state=0)

reg_sm3 = sm.OLS(y_train3, X_train3).fit()
reg_sm3.summary()

y_hat3 = reg_sm3.predict(X_test3)

rmse3 = math.sqrt(mean_squared_error(y_hat3,y_test3))

plt.scatter(y_hat3,y_test3)

#cross validation 5 fold 
from sklearn.model_selection import cross_val_score 
X4 = pd.get_dummies(df_filter[['mag','fat','loss','len','st','year']])
y4 = df_filter.inj.values

X_train4, X_test4, y_train4, y_test4 = train_test_split(X4, y4, test_size=0.3, random_state=0)

reg4 = LinearRegression().fit(X_train4, y_train4)
reg4.score(X_train4,y_train4)

scores = cross_val_score(reg4,X4,y4, cv=5, scoring = 'neg_mean_squared_error')
np.sqrt(np.abs(scores))





