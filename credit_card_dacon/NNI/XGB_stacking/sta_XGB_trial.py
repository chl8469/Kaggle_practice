from os.path import join
from os import remove as remove_file
import glob
import pickle
import multiprocessing

n_cpus = multiprocessing.cpu_count()

import nni
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

BASE_DIR = '/Users/HwaLang/Desktop/python/T academy/Kaggle_camp/'

scaler_dict = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler   
}

def select_scaler(name):
    print(f"Select {name} Scaler")
    
    return scaler_dict[name]()

def preprocess(x_train, x_valid, x_test, params):
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()    
    tmp_x_test  = x_test.copy()
    
    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)
    
    scaler = select_scaler(params['scaler'])
    tmp_x_train = scaler.fit_transform(tmp_x_train)
    tmp_x_valid = scaler.transform(tmp_x_valid)
    tmp_x_test  = scaler.transform(tmp_x_test)

    return tmp_x_train, tmp_x_valid, tmp_x_test

def make_submission(y_test_pred, val_loss):
    ''' 
        내용 추가
        제출 파일 생성 함수 작성
    '''
    submit_path = join(BASE_DIR, 'data', 'MDC14', 'sample_submission.csv')
    df_result = pd.read_csv(submit_path)
    df_result.iloc[:, 1:] = y_test_pred
    df_result.to_csv('./submit/model_loss_{:.4f}.csv'.format(val_loss), index=False)

    pass

def save_model(model, filename):
    with open(filename, "wb") as f:
        pickle.dump(model, f)

def main(params):
    global num_columns, cat_columns

    # 데이터 경로 입력
    train_path = join(BASE_DIR, 'credit_card_dacon', 'stacked_X.csv')
    label_path = join(BASE_DIR, 'credit_card_dacon', 'stacked_y.csv')
    test_path  = join(BASE_DIR, 'credit_card_dacon', 'test_ensem.csv')

    data = pd.read_csv(train_path)
    label = pd.read_csv(label_path)
    test = pd.read_csv(test_path)

    # data.drop(columns=['index', 'credit'], inplace=True)
    # test.drop(columns=['index'], inplace=True) 

    # cat_columns = [c for (c, t) in zip(data.dtypes.index, data.dtypes) if t == 'O'] 
    # num_columns = [c for c in data.columns if c not in cat_columns]

    le = LabelEncoder()
    label = le.fit_transform(label)
    
    # 대회에 맞는 차원 입력
    y_test_pred = np.zeros((test.shape[0], le.classes_.shape[0]))

    cv_scores = list()

    # 예측하려는 유형에 따라 KFold or StratifiedKFold 선택
    n_splits = 5
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for train_index, valid_index in skf.split(data, label):
        x_train, y_train = data.iloc[train_index, :], label[train_index]
        x_valid, y_valid = data.iloc[valid_index, :], label[valid_index]

        x_train, x_valid, x_test = preprocess(x_train, x_valid, test, params)

        model = XGBClassifier(n_estimators     = 1000000,
                              subsample        = params['subsample'],
                              max_depth        = params['max_depth'],
                              colsample_bytree = params['colsample_bytree'],
                              eta              = params['eta'],
                              n_jobs           = 4)
        
        model.fit(x_train, y_train,
                  eval_set=[[x_train, y_train], [x_valid, y_valid]],
                  eval_metric='mlogloss',
                  early_stopping_rounds=200,
                  verbose=100)

        # train_loss = log_loss(y_train, model.predict_proba(x_train))
        valid_loss = log_loss(y_valid, model.predict_proba(x_valid))

        cv_scores.append(valid_loss)

        y_test_pred += model.predict_proba(x_test) / skf.n_splits

    cv_loss = np.mean(cv_scores)

    print('Cross validation Loss: %.4f' % cv_loss)

    nni.report_final_result(cv_loss)
    print('Final result is %g', cv_loss)
    print('Send final result done.')

    make_submission(y_test_pred, cv_loss)

    global best_model
    best_model = glob.glob("./model/*.pkl")
    
    is_model = int(best_model[0].split('.')[-2]) if best_model else 10

    if is_model > cv_loss:
        save_model(model, f"./model/XGB_model{cv_loss:.4f}.pkl")

if __name__ == '__main__':

    best_model = glob.glob("./model/*.pkl")
    best_model.sort()
    if len(best_model) > 3:
        try:
            [remove_file(_) for _ in best_model[3:]]
        except Exception as e:
            print(e)

    params = nni.get_next_parameter()
    main(params)