# -*- coding: utf-8 -*-

# -- Sheet --

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
plt.style.use('seaborn')
sns.set(font_scale=1.8)
import random
import warnings
warnings.filterwarnings('ignore')

app_df = pd.read_csv('AppleStore.csv')
app_df.head()

app_df.info()

app_df = app_df.iloc[:, 1:]

app_df['size_MB']= app_df['size_bytes']/(1024*1024)
app_df['is_free']=app_df['price'].apply(lambda x: 0 if x>0 else 1)
app_df.nunique()

#due to currency has only USD, it can be dropped
app_df= app_df.drop(['currency'], axis =1)

#plot free vs.non-free app
app_df['is_free'].value_counts().plot.bar()
plt.title('Free app vs Non-Free app')
plt.show()

fig, ax = plt.subplots()
genre_df = app_df['prime_genre'].value_counts()
sns.barplot(genre_df.index, genre_df)
ax.set_xticklabels(genre_df.index, rotation='90')
plt.show()

genre_free = app_df.groupby('is_free')['prime_genre'].value_counts().sort_index().unstack().T.sort_values(by = 0, ascending = False)
print(genre_free)

fig, ax = plt.subplots()
ax.plot(genre_free.index, genre_free[1], color = 'red')
ax.plot(genre_free.index, genre_free[0], color = 'blue')
ax.set_xticklabels(genre_free.index, rotation = '90')
ax.legend(['is free','non_free'], loc ='upper right')
plt.show()

# Looking for non_free social network app
social_app = app_df[app_df['is_free']==0]
social_app= social_app[social_app['prime_genre']=='Social Networking']
social_app.head()
social_app['track_name'].value_counts()

genre_rate = app_df.groupby('prime_genre')['user_rating'].mean().sort_values(ascending = False).plot.bar()

app_df=app_df.iloc[:,1:]

app_df.head()

#correlation
app_corr = app_df.corr()
sns.heatmap(app_corr)
plt.show()

rating_corr=app_corr['user_rating'].sort_values(ascending=False)
price_corr = app_corr['price'].sort_values(ascending=False)
plt.scatter(rating_corr,price_corr)
plt.show()

rating_corr=app_corr['user_rating'].sort_values(ascending=False)
rating_corr

from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import xgboost as xgb

app_df['rating_count_before']= app_df['rating_count_tot']-app_df['rating_count_ver']

app_df.nunique()

app_df.dtypes

labelencode = LabelEncoder()
train_df = app_df.copy()
#train_df['ver']= onehotcode.fit_transform(train_df['ver'])
train_df['cont_rating']= labelencode.fit_transform(train_df['cont_rating'])
train_df['prime_genre']=labelencode.fit_transform(train_df['prime_genre'])

train_df=train_df.drop(['ver','track_name'], axis=1)

target = train_df['user_rating']
def cate_rating(x):
    if x<=4:
        return 0
    else:
        return 1
    
target = target.apply(cate_rating)

df_train = app_df[['size_MB', 'is_free', 'price', 'rating_count_before', 'sup_devices.num', 'ipadSc_urls.num', 'lang.num', 'vpp_lic', 'prime_genre']]
df_train=pd.get_dummies(df_train)

#train_df=train_df.drop(['sup_devices.num','ipadSc_urls.num','vpp_lic','lang.num'], axis =1)

'''
train_df['price']=train_df['price'].astype(int)
train_df['user_rating_ver']=train_df['user_rating_ver'].astype(int)
train_df['size_MB']=train_df['size_MB'].astype(int)
'''

X_train,X_test,y_train,y_test = train_test_split(df_train, target, train_size =0.6, random_state=42)

from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score, cross_validate
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

'''rf = RandomForestClassifier()
kfold = KFold(n_splits=5)
result = cross_validate(rf,X_train,y_train,cv=kfold)
print(result)'''

models = [RandomForestClassifier(), XGBClassifier()]

kfold = KFold(n_splits=5)

clf_comparison = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score'])

for i, model in enumerate(models):
    clf = model
    cv_result = cross_validate(model, X_train, y_train, cv=kfold, scoring='accuracy', return_train_score=True)
    clf_comparison.loc[i, 'Classfier_name'] = model.__class__.__name__
    clf_comparison.loc[i, 'train_score'] = cv_result['train_score'].mean()
    clf_comparison.loc[i, 'test_score'] = cv_result['test_score'].mean()

clf_comparison

print('----------for val dataset---------')

models_test = [RandomForestClassifier(), XGBClassifier()]

kfold = KFold(n_splits=5)

clf_comparison_test = pd.DataFrame(columns=['Classfier_name', 'train_score', 'test_score'])

for i, model in enumerate(models_test):
    clf = model
    cv_result = cross_validate(model, X_test, y_test, cv=kfold, scoring='accuracy', return_train_score=True)
    clf_comparison_test.loc[i, 'Classfier_name'] = model.__class__.__name__
    clf_comparison_test.loc[i, 'train_score'] = cv_result['train_score'].mean()
    clf_comparison_test.loc[i, 'test_score'] = cv_result['test_score'].mean()

clf_comparison_test

