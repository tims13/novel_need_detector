import numpy as np
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.metrics import recall_score, precision_score, accuracy_score
from config import *

def evaluate_novel(c, model, train_features, test_features, y_true, name):
    model.fit(train_features)
    y_pred = model.predict(test_features)
    y_pred[y_pred != -1] = 0
    y_pred[y_pred == -1] = 1
    rec_score = recall_score(y_true, y_pred)
    prec_score = precision_score(y_true, y_pred)
    acc_score = accuracy_score(y_true, y_pred)
    print('-------------------------------')
    print(name + ','+ str(c))
    print('RECALL:' + str(rec_score))
    print('PRECISION:' + str(prec_score))
    print('ACCURACY:' + str(acc_score))
    return y_pred

data_need_detected = pd.read_csv(data_need_detected_path)
y_true = data_need_detected['novel']
y_true = np.array(y_true, np.int)

data = np.load(data_np_path + '.npz')
train = data['train']
test = data['test']

print('Train: ' + str(train.shape))
print('Test: ' + str(test.shape))

list_c = np.linspace(0.05,0.2,11,endpoint=True)
print('LOF Training...')
for c in list_c:
    model = LocalOutlierFactor(n_neighbors=20, contamination=c, novelty=True, n_jobs=-1)
    y_pred = evaluate_novel(c, model, train, test, y_true, 'LOF')
    data_need_detected['LOF,c=' + str(c)] = y_pred

print('Iso-Forest Training...')
for c in list_c:
    model = IsolationForest(random_state=0, contamination=c, n_jobs=-1)
    y_pred = evaluate_novel(c, model, train, test, y_true, 'Iso-Forest')
    data_need_detected['ISO,c=' + str(c)] = y_pred

print('OneClassSVM Training...')
model = OneClassSVM(gamma='auto')
y_pred = evaluate_novel(c, model, train, test, y_true, 'OneClass-SVM')
data_need_detected['SVM,c=' + str(c)] = y_pred

data_need_detected.to_csv(data_result_csv_path)