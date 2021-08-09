from os.path import join
import multiprocessing

n_cpus = multiprocessing.cpu_count()

import nni
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

BASE_DIR = '../'

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
    
    occyp_mode = tmp_x_train['occyp_type'].mode()[0]
    
    tmp_x_train.loc[pd.isna(tmp_x_train['occyp_type']), 'occyp_type'] = occyp_mode
    tmp_x_valid.loc[pd.isna(tmp_x_valid['occyp_type']), 'occyp_type'] = occyp_mode
    tmp_x_test.loc[pd.isna(tmp_x_test['occyp_type'])  , 'occyp_type'] = occyp_mode
    
    scaler = select_scaler(params['scaler'])
    tmp_x_train[num_columns] = scaler.fit_transform(tmp_x_train[num_columns])
    tmp_x_valid[num_columns] = scaler.transform(tmp_x_valid[num_columns])
    tmp_x_test[num_columns]  = scaler.transform(tmp_x_test[num_columns])
    
    x_all = pd.concat([tmp_x_train[cat_columns], 
                       tmp_x_valid[cat_columns], 
                       tmp_x_test[cat_columns]], axis=0)
    
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(x_all)
    
    ohe_columns = list()
    for cat_cols in ohe.categories_:
        ohe_columns += cat_cols.tolist()
    
    new_cat_train = pd.DataFrame(ohe.transform(tmp_x_train[cat_columns]), columns=ohe_columns)
    new_cat_valid = pd.DataFrame(ohe.transform(tmp_x_valid[cat_columns]), columns=ohe_columns)
    new_cat_test  = pd.DataFrame(ohe.transform(tmp_x_test[cat_columns]),  columns=ohe_columns)
    
    tmp_x_train.drop(columns=cat_columns, inplace=True)
    tmp_x_valid.drop(columns=cat_columns, inplace=True)
    tmp_x_test.drop(columns=cat_columns, inplace=True)
    
    tmp_x_train = pd.concat([tmp_x_train, new_cat_train], axis=1)
    tmp_x_valid = pd.concat([tmp_x_valid, new_cat_valid], axis=1)
    tmp_x_test  = pd.concat([tmp_x_test,  new_cat_test],  axis=1)
    
    return tmp_x_train.values, tmp_x_valid.values, tmp_x_test.values

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

def main(params):
    global num_columns, cat_columns

    # 데이터 경로 입력
    train_path = join(BASE_DIR, 'data', 'MDC14', 'train.csv')
    test_path  = join(BASE_DIR, 'data', 'MDC14', 'test.csv')

    data = pd.read_csv(train_path)
    test = pd.read_csv(test_path)

    label = data['credit'] 

    data.drop(columns=['index', 'credit'], inplace=True)
    test.drop(columns=['index'], inplace=True) 

    cat_columns = [c for (c, t) in zip(data.dtypes.index, data.dtypes) if t == 'O'] 
    num_columns = [c for c in data.columns if c not in cat_columns]

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

        train_loss = log_loss(y_train, model.predict_proba(x_train))
        valid_loss = log_loss(y_valid, model.predict_proba(x_valid))

        cv_scores.append(valid_loss)

        y_test_pred += model.predict_proba(x_test) / skf.n_splits

    cv_loss = np.mean(cv_scores)

    print('Cross validation Loss: %.4f' % cv_loss)

    nni.report_final_result(cv_loss)
    print('Final result is %g', cv_loss)
    print('Send final result done.')

    make_submission(y_test_pred, cv_loss)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)