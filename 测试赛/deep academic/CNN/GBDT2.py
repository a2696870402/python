#
# 预测版块
#

import numpy as np
import pandas as pd
from sklearn.externals import joblib
from sklearn import preprocessing
from sklearn import metrics
from sklearn.preprocessing import StandardScaler

standarscaler = StandardScaler()


def return_test_data():
    # 预测数据的预处理
    data = pd.read_csv("data/first_round_testing_data.csv")
    data_X = data[['Parameter1', 'Parameter2', 'Parameter3',
                   'Parameter4', 'Parameter5', 'Parameter6',
                   'Parameter7', 'Parameter8', 'Parameter9', 'Parameter10']]
    data_X = standarscaler.fit_transform(data_X)
    return data_X


clf = joblib.load('train_model_result.m')
x_test_data = return_test_data()
y_test_data = clf.predict(x_test_data)
predict_es = np.asarray(clf.predict_proba(x_test_data))
print(predict_es)
data1 = pd.DataFrame(predict_es)
data1.to_csv('jieguo/gailv.csv')
# with open("jieguo/yuce.csv", "w") as f:
#     for i in range(len(y_test_data)):
#         f.write(str(i+1)+","+str(int(y_test_data[i]))+"\n")
# print("predict over!!!")
# print("y_predict=", y_test_data)

