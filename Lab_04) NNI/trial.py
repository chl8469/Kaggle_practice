from os.path import join
import multiprocessing

n_cpus = multiprocessing.cpu_count()

import nni
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from sklearn.metrics import log_loss
from xgboost import XGBClassifier

BASE_DIR = 'BASE 경로 입력'

def preprocess(x_train, x_valid, x_test):
    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()    
    tmp_x_test  = x_test.copy()
    
    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)
    
    # 전처리 함수 작성
    
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

    
    
    # 대회에 맞는 차원 입력
    y_test_pred = np.zeros()

    cv_scores = list()

    # 예측하려는 유형에 따라 KFold or StratifiedKFold 선택
    n_splits = 5
    kf = 
    
    for trn_idx, val_idx in kf.split(data, label):
        x_train, y_train = 
        x_valid, y_valid = 
        
        # 전처리
        x_train, x_valid, x_test = 

        # 모델 정의 및 파라미터 전달
        model = 
        
        # 모델 학습 및 Early Stopping 적용
        model.fit()

        # Loss 계산
        train_loss = 
        valid_loss = 

        cv_scores.append(valid_loss)

        y_test_pred = model.predict_proba(x_test) / n_splits

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