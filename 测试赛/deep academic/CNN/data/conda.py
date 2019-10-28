# -*- coding: utf-8 -*-
"""
Created on Sat Aug 24 22:36:31 2019

@author: 31037
"""


import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
from sklearn import preprocessing
import lightgbm as lgb
from sklearn import metrics,ensemble
from sklearn.model_selection import RepeatedStratifiedKFold,KFold,StratifiedKFold
#from sklearn import neural_network
from sklearn.decomposition import PCA,FactorAnalysis,TruncatedSVD
from imblearn.over_sampling import RandomOverSampler
warnings.filterwarnings('ignore')


PATH = ''
test = pd.read_csv(PATH + 'first_round_testing_data.csv')
train = pd.read_csv(PATH + 'first_round_training_data.csv')
# train['Group'] = test['Group']
sub_group = pd.DataFrame(test['Group'])
subsample = pd.read_csv(PATH + 'submit_example.csv')

features = ['Parameter1', 'Parameter2', 'Parameter3', 'Parameter4', 'Parameter5',
            'Parameter6', 'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']
target = train['Quality_label']

lbl = preprocessing.LabelEncoder()
lbl.fit(target)
target = pd.Series(lbl.transform(target))


data = pd.concat([train[features], test[features]], ignore_index=True)
# features.remove('Group')

n_unique = [data[col].nunique() for col in features]
n_unique = pd.DataFrame(n_unique, index=features, columns=['nunique'])
plt.figure(figsize=(8,6))
sns.barplot(x=n_unique.index, y=n_unique['nunique'])
plt.show()

category_cols = ['Parameter'+str(i) for i in range(5,11)]
num_cols = ['Parameter'+str(i) for i in range(1,5)]
for c_col in category_cols:
    # lbl = preprocessing.LabelEncoder()
    # data[c_col] = lbl.fit_transform(data[c_col])
    data[c_col+'_cnt'] = data[c_col].map(data[c_col].value_counts())
    for n_col in num_cols:
        if n_col+'_log' not in data.columns.values:
            data[n_col+'_log'] = np.log(data[n_col])
        data[n_col+'_groupby_'+c_col+'_mean_ratio'] = \
            data[n_col] / (data[c_col].map(data[n_col].groupby(data[c_col]).mean()))


def _std_bins(x, s, m):
    if x > m + 3*s or x < m - 3*s:
        return 3
    elif x > m + 2*s or x < m - 2*s:
        return 2
    elif x > m + s or x < m - s:
        return 1
    else:
        return 0


bin_cols = [col+str('_log') for col in num_cols]
for col in bin_cols:
    std = data[col].std()
    mean = data[col].mean()
    data[col+'_std_bins'] = data[col].apply(lambda x: _std_bins(x, std, mean))
    data[col+'_std_ratio'] = data[col] / data[col].std()

data['std_num_cols'] = data[num_cols].std(axis=1)
data['mean_num_cols'] = data[num_cols].mean(axis=1)

'''
n_com = 2
tsvd = TruncatedSVD(n_components=n_com,random_state=12)
for i in range(n_com):
    data['tsvd'+str(i)]=tsvd.fit_transform(data[features])[:,i]
'''

# Parameter5, Parameter6, Parameter7, Parameter8, Parameter9, Parameter10是类别变量
# data.drop(['Group'],axis=1,inplace=True)

train = data.iloc[0:train.shape[0], ]
# features = train.columns.values
# ros = RandomOverSampler(random_state=128)
# train,target = ros.fit_resample(train,target)
# train = pd.DataFrame(train,columns=features)
# target = pd.Series(target,name='target')

test = data.iloc[train.shape[0]:,]
del data

# folds = KFold(n_splits=5,shuffle=True,random_state=128)
folds = StratifiedKFold(n_splits=5, random_state=12)
prediction = np.zeros([test.shape[0],4])
oof = np.zeros([target.shape[0],4])

feature_importance = pd.DataFrame(train.columns.values,columns=['features'])
feature_importance['importance'] = np.zeros(feature_importance.shape[0])

params = {'num_leaves': 256,
          'feature_fraction': 0.6,
          'bagging_fraction': 0.4,
          'min_data_in_leaf': 32,
          'objective': 'multiclass',
          'num_class':4,
          'learning_rate': 0.005,
          "boosting_type": "gbdt",
          "bagging_seed": 11,
          "metric": 'multi_logloss',
          "verbosity": -1,
          'lambda_l1': 0.35,
          'lambda_l2': 0.45,
          'random_state': 47,
          'is_unbalance':True
         }
seeds_num = 5
seeds = np.random.randint(100, 1000, seeds_num)
epoch = 0

for seed in seeds:
    epoch += 1
    print(f'----------seed:{seed}',f'epoch:{epoch}------------')
    for fold_, (trn_idx, val_idx) in enumerate(folds.split(train, target)):
        print("Fold {}".format(fold_))
        clf = lgb.LGBMClassifier(**params, n_estimators=3000)
        clf.fit(train.iloc[trn_idx], target.iloc[trn_idx], eval_set = [(train.iloc[val_idx], target.iloc[val_idx])], verbose=500,early_stopping_rounds=200) #, early_stopping_rounds=100
        oof[val_idx] += clf.predict_proba(train.iloc[val_idx])
        # oof += clf.predict_proba(train)
        prediction += clf.predict_proba(test) / folds.n_splits
        feature_importance['importance'] += clf.feature_importances_

oof = oof / seeds_num
prediction = prediction / seeds_num
print('log_loss:{}'.format(metrics.log_loss(target,oof)))


def _softmax(matrix):
    matrix_sort = np.argmax(matrix, axis=1)
    # matrix_sort = np.argmax(np.bincount(matrix))
    return matrix_sort


oof = _softmax(oof)
print(metrics.classification_report(target,oof))

feature_importance.to_excel(PATH + 'feature_importance.xlsx', encoding='utf-8-sig')

plt.figure(figsize=(8, 20))
sns.barplot(x='importance',y='features',data=feature_importance.sort_values('importance',ascending=False))
plt.title('feature_importance',fontsize=20)
plt.show()

cols = ['Excellent', 'Fail', 'Good', 'Pass']
# 0,1,2,3
# prediction = prediction / (fold_+1)
sub = pd.DataFrame(prediction, columns=list(lbl.classes_))
sub['Group'] = sub_group
sub = sub.groupby('Group')[cols].mean()
sub.columns = [col+' ratio' for col in sub.columns.values]
sub.reset_index(inplace=True)
sub = sub[subsample.columns]
import datetime
filename="{:%Y-%m-%d_%H_%M}_sub".format(datetime.datetime.now())
sub.to_csv(PATH + filename + '_accuracy{}.csv'.format(round(metrics.accuracy_score(target,oof),4)), index=False)
