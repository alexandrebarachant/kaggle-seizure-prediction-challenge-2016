import numpy as np
import json
import os
from scipy.io import loadmat
from pandas import DataFrame

import xgboost as xgb
from config_name_creator import create_fft_data_name
##xgb
def load_train_data_xgb(data_path, subject):
    read_dir = data_path + '/' + subject
    filenames = sorted(os.listdir(read_dir))
    train_filenames = []
    for filename in filenames:
        train_filenames.append(filename)
    n = len(train_filenames)
    datum = loadmat(read_dir + '/' + train_filenames[1], squeeze_me=True)
    x = np.zeros(((n,) + datum['data'].shape), dtype='float32')
    y = np.zeros(n, dtype='int8')

    filename_to_idx = {}
    for i, filename in enumerate(train_filenames):
        datum = loadmat(read_dir + '/' + filename, squeeze_me=True)
        x[i] = datum['data']
        y[i] = 1 if filename.endswith('_1.mat') else 0
        filename_to_idx[subject + '/' + filename] = i

    return {'x': x, 'y': y, 'filename_to_idx': filename_to_idx}

def load_test_data(data_path, subject):
    read_dir = data_path + '/' + subject
    data, id = [], []
    filenames = sorted(os.listdir(read_dir))
    for filename in filenames:
        data.append(loadmat(read_dir + '/' + filename, squeeze_me=True))
        id.append(filename)
    n_test = len(data)
    x = np.zeros(((n_test,) + data[0]['data'].shape), dtype='float32')
    for i, datum in enumerate(data):
        x[i] = datum['data']

    return {'x': x, 'id': id}

def reshape_data(x, y=None):
    n_examples = x.shape[0]
    n_channels = x.shape[1]
    n_fbins = x.shape[2]
    n_timesteps = x.shape[3]
    x_new = np.zeros((n_examples * n_timesteps, n_channels, n_fbins))
    for i in range(n_channels):
        xi = np.transpose(x[:, i, :, :], axes=(0, 2, 1))
        xi = xi.reshape((n_examples * n_timesteps, n_fbins))
        x_new[:, i, :] = xi

    x_new = x_new.reshape((n_examples * n_timesteps, n_channels * n_fbins))
    if y is not None:
        y_new = np.repeat(y, n_timesteps) ## expanding the sample size
        return x_new, y_new
    else:
        return x_new

params = {
        "objective": "binary:logistic",
        "booster" : "gbtree",
        "eval_metric": "auc",
        "eta": 0.22,##0.22
        "max_depth": 3,
        "subsample": 0.80,
        "colsample_bytree": 0.78,
        "silent": 1,
    }

def train(subject,data_path,params):
    d=load_train_data_xgb(data_path,subject)
    x,y=reshape_data(d['x'],d['y'])
    dtrain=xgb.DMatrix(x,y)
    gbm=xgb.train(params,dtrain,num_boost_round=500,verbose_eval=20)
    return gbm

def predict(subject,data_path,model):
    dtest=load_test_data(data_path,subject)
    x_test, id = dtest['x'], dtest['id']
    n_test_examples = x_test.shape[0]
    n_timesteps = x_test.shape[3]

    x_test = reshape_data(x_test)
    test=xgb.DMatrix(x_test)

    pred_1m=model.predict(test)
    pred_10m = np.reshape(pred_1m, (n_test_examples, n_timesteps))
    pred_10m = np.mean(pred_10m, axis=1)
    pred_list = zip(id, pred_10m)
    return pred_list


#data path
with open('kaggle_SETTINGS.json') as f:
    settings_dict = json.load(f)
data_path= settings_dict['path']['processed_data_path'] + '/'+ create_fft_data_name(settings_dict)
submission_path=settings_dict['path']['submission_path']+'/'


def get_prediction(data_path):
    train_subjects=['train_1','train_2','train_3']
    test_subjects=['test_1_new','test_2_new','test_3_new']
    pred=[]
    for i in range(3):
        gbm=train(train_subjects[i],data_path,params)

        singpred=predict(test_subjects[i],data_path,model=gbm)
        pred=pred+singpred
    df = DataFrame(data=pred, columns=['File', 'Class'])
    return df
#generate dataframe
df=get_prediction(data_path)
df.to_csv(submission_path+"Feng_xgb.csv",index=False, header=True)
