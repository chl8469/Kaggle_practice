from os.path import join
import multiprocessing

n_cpus = multiprocessing.cpu_count()

import nni
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier

BASE_DIR = '../'

def preprocess(x_train, x_valid, x_test):
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()    
    tmp_x_test  = x_test.copy()
    
    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)
    
    # 결측치 처리
    imputer = SimpleImputer(strategy='most_frequent')
    tmp_x_train[cat_columns] = imputer.fit_transform(tmp_x_train[cat_columns])
    tmp_x_valid[cat_columns] = imputer.transform(tmp_x_valid[cat_columns])
    tmp_x_test[cat_columns]  = imputer.transform(tmp_x_test[cat_columns])
    
    # 스케일링
    scaler = StandardScaler()
    tmp_x_train[num_columns] = scaler.fit_transform(tmp_x_train[num_columns])
    tmp_x_valid[num_columns] = scaler.transform(tmp_x_valid[num_columns])
    tmp_x_test[num_columns]  = scaler.transform(tmp_x_test[num_columns])

    # 인코딩
    ohe = OneHotEncoder(sparse=False)
    ohe.fit(tmp_x_train[cat_columns])
    
    tmp_x_train_cat = pd.DataFrame(ohe.transform(tmp_x_train[cat_columns]))
    tmp_x_valid_cat = pd.DataFrame(ohe.transform(tmp_x_valid[cat_columns]))
    tmp_x_test_cat  = pd.DataFrame(ohe.transform(tmp_x_test[cat_columns]))
    
    tmp_x_train.drop(columns=cat_columns, inplace=True)
    tmp_x_valid.drop(columns=cat_columns, inplace=True)
    tmp_x_test.drop(columns=cat_columns, inplace=True)
    
    tmp_x_train = pd.concat([tmp_x_train, tmp_x_train_cat], axis=1)
    tmp_x_valid = pd.concat([tmp_x_valid, tmp_x_valid_cat], axis=1)
    tmp_x_test  = pd.concat([tmp_x_test, tmp_x_test_cat], axis=1)
    
    return tmp_x_train.values, tmp_x_valid.values, tmp_x_test.values

def make_submission(y_test_pred, val_loss):
    ''' 
        내용 추가
        제출 파일 생성 함수 작성
    '''

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
    test.drop(columns=['index'],           inplace=True)

    cat_columns = [c for c, t in zip(data.dtypes.index, data.dtypes) if t == 'O'] 
    num_columns = [c for c    in data.columns if c not in cat_columns]

    
    # 대회에 맞는 차원 입력
    y_test_pred = np.zeros((test.shape[0], 3))

    cv_scores = list()

    # 예측하려는 유형에 따라 KFold or StratifiedKFold 선택
    n_splits = 5
    kf = StratifiedKFold()
    
    for trn_idx, val_idx in kf.split(data, label):
        x_train, y_train = data.iloc[trn_idx, :], label.iloc[trn_idx,]
        x_valid, y_valid = data.iloc[val_idx, :], label.iloc[val_idx,]
        
        # 전처리
        x_train, x_valid, x_test = preprocess(x_train, x_valid, test)

        # 모델 정의 및 파라미터 전달
        model = XGBClassifier()
        
        # 모델 학습 및 Early Stopping 적용
        model.fit(x_train, y_train)

        # Loss 계산
        train_loss = log_loss(y_train, model.predict_proba(x_train))
        valid_loss = log_loss(y_valid, model.predict_proba(x_valid))

        cv_scores.append(valid_loss)

        y_test_pred += model.predict_proba(x_test) / n_splits

    cv_loss = np.mean(cv_scores)

    print('Cross validation Loss: %.4f' % cv_loss)
    
    # 학습 결과 리포팅
    nni.report_final_result(cv_loss)
    print('Final result is %g', cv_loss)
    print('Send final result done.')

    make_submission(y_test_pred, cv_loss)

if __name__ == '__main__':
    params = nni.get_next_parameter()
    main(params)