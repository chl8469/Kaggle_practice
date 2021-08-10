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
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler, PowerTransformer
from sklearn.metrics import log_loss
from lightgbm import LGBMClassifier
from category_encoders.ordinal import OrdinalEncoder

BASE_DIR = '/Users/HwaLang/Desktop/python/T academy/Kaggle_camp/'

scaler_dict = {
    "minmax": MinMaxScaler,
    "standard": StandardScaler   
}

def select_scaler(name):
    print(f"Select {name} Scaler")
    
    return scaler_dict[name]()

def preprocess(x_train, x_valid, x_test, params, target):
    # global num_columns, cat_columns

    tmp_x_train = x_train.copy()
    tmp_x_valid = x_valid.copy()    
    tmp_x_test  = x_test.copy()
# ---------------------------
    tmp_x_train.drop(columns = ['FLAG_MOBIL'], inplace = True)
    tmp_x_valid.drop(columns = ['FLAG_MOBIL'], inplace = True)
    tmp_x_test.drop(columns = ['FLAG_MOBIL'], inplace = True)

    tmp_x_train['DAYS_EMPLOYED'] = tmp_x_train['DAYS_EMPLOYED'].map(lambda x: 0 if x > 0 else x)
    tmp_x_valid['DAYS_EMPLOYED'] = tmp_x_valid['DAYS_EMPLOYED'].map(lambda x: 0 if x > 0 else x)
    tmp_x_test['DAYS_EMPLOYED'] = tmp_x_test['DAYS_EMPLOYED'].map(lambda x: 0 if x > 0 else x)

    feats = ['DAYS_BIRTH', 'begin_month', 'DAYS_EMPLOYED']
    for feat in feats:
        tmp_x_train[feat]=np.abs(tmp_x_train[feat])
        tmp_x_valid[feat]=np.abs(tmp_x_valid[feat])
        tmp_x_test[feat]=np.abs(tmp_x_test[feat])
    
    for df in [tmp_x_train, tmp_x_valid, tmp_x_test]:
        # before_EMPLOYED: 고용되기 전까지의 일수
        df['before_EMPLOYED'] = df['DAYS_BIRTH'] - df['DAYS_EMPLOYED']
        df['income_total_befofeEMP_ratio'] = df['income_total'] / df['before_EMPLOYED']
        df['before_EMPLOYED_m'] = np.floor(df['before_EMPLOYED'] / 30) - ((np.floor(df['before_EMPLOYED'] / 30) / 12).astype(int) * 12)
        df['before_EMPLOYED_w'] = np.floor(df['before_EMPLOYED'] / 7) - ((np.floor(df['before_EMPLOYED'] / 7) / 4).astype(int) * 4)
        
        #DAYS_BIRTH 파생변수- Age(나이), 태어난 월, 태어난 주(출생연도의 n주차)
        df['Age'] = df['DAYS_BIRTH'] // 365
        df['DAYS_BIRTH_m'] = np.floor(df['DAYS_BIRTH'] / 30) - ((np.floor(df['DAYS_BIRTH'] / 30) / 12).astype(int) * 12)
        df['DAYS_BIRTH_w'] = np.floor(df['DAYS_BIRTH'] / 7) - ((np.floor(df['DAYS_BIRTH'] / 7) / 4).astype(int) * 4)

        
        #DAYS_EMPLOYED_m 파생변수- EMPLOYED(근속연수), DAYS_EMPLOYED_m(고용된 달) ,DAYS_EMPLOYED_w(고용된 주(고용연도의 n주차))  
        df['EMPLOYED'] = df['DAYS_EMPLOYED'] // 365
        df['DAYS_EMPLOYED_m'] = np.floor(df['DAYS_EMPLOYED'] / 30) - ((np.floor(df['DAYS_EMPLOYED'] / 30) / 12).astype(int) * 12)
        df['DAYS_EMPLOYED_w'] = np.floor(df['DAYS_EMPLOYED'] / 7) - ((np.floor(df['DAYS_EMPLOYED'] / 7) / 4).astype(int) * 4)

        #ability: 소득/(살아온 일수+ 근무일수)
        df['ability'] = df['income_total'] / (df['DAYS_BIRTH'] + df['DAYS_EMPLOYED'])
        
        #income_mean: 소득/ 가족 수
        df['income_mean'] = df['income_total'] / df['family_size']
        
        #ID 생성: 각 컬럼의 값들을 더해서 고유한 사람을 파악(*한 사람이 여러 개 카드를 만들 가능성을 고려해 begin_month는 제외함)
        df['ID'] = \
        df['child_num'].astype(str) + '_' + df['income_total'].astype(str) + '_' +\
        df['DAYS_BIRTH'].astype(str) + '_' + df['DAYS_EMPLOYED'].astype(str) + '_' +\
        df['work_phone'].astype(str) + '_' + df['phone'].astype(str) + '_' +\
        df['email'].astype(str) + '_' + df['family_size'].astype(str) + '_' +\
        df['gender'].astype(str) + '_' + df['car'].astype(str) + '_' +\
        df['reality'].astype(str) + '_' + df['income_type'].astype(str) + '_' +\
        df['edu_type'].astype(str) + '_' + df['family_type'].astype(str) + '_' +\
        df['house_type'].astype(str) + '_' + df['occyp_type'].astype(str)

    cols = ['child_num', 'DAYS_BIRTH', 'DAYS_EMPLOYED',]
    tmp_x_train.drop(cols, axis=1, inplace=True)
    tmp_x_valid.drop(cols, axis=1, inplace=True)
    tmp_x_test.drop(cols, axis=1, inplace=True)

    cat_columns = [c for (c, t) in zip(tmp_x_train.dtypes.index, tmp_x_train.dtypes) if t == 'O'] 
    num_columns = [c for c in tmp_x_train.columns if c not in cat_columns]

    YJ_transform = PowerTransformer(method='yeo-johnson')
    tmp_x_train['income_total'] = YJ_transform.fit_transform(tmp_x_train['income_total'].values.reshape(-1, 1))
    tmp_x_valid['income_total'] = YJ_transform.transform(tmp_x_valid['income_total'].values.reshape(-1, 1))
    tmp_x_test['income_total'] = YJ_transform.transform(tmp_x_test['income_total'].values.reshape(-1, 1))


    tmp_x_train.reset_index(drop=True, inplace=True)
    tmp_x_valid.reset_index(drop=True, inplace=True)
    
    num_columns.remove("income_total")

    scaler = select_scaler(params['scaler'])
    tmp_x_train[num_columns] = scaler.fit_transform(tmp_x_train[num_columns])
    tmp_x_valid[num_columns] = scaler.transform(tmp_x_valid[num_columns])
    tmp_x_test[num_columns]  = scaler.transform(tmp_x_test[num_columns])

    ode = OrdinalEncoder(cat_columns)
    tmp_x_train[cat_columns] = ode.fit_transform(tmp_x_train[cat_columns], target)
    tmp_x_valid[cat_columns] = ode.transform(tmp_x_valid[cat_columns])
    tmp_x_test[cat_columns]  = ode.transform(tmp_x_test[cat_columns])

    tmp_x_train['ID'] = tmp_x_train['ID'].astype('int64')
    tmp_x_valid['ID'] = tmp_x_valid['ID'].astype('int64')
    tmp_x_test['ID'] = tmp_x_test['ID'].astype('int64')

# ---------------------------    
    return tmp_x_train.values, tmp_x_valid.values, tmp_x_test

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

        x_train, x_valid, x_test = preprocess(x_train, x_valid, test, params, y_train)

        model = LGBMClassifier(n_estimators     = 1000000,
                               subsample        = params['subsample'],
                               max_depth        = params['max_depth'],
                               colsample_bytree = params['colsample_bytree'],
                               learning_rate    = params['lr'],
                               boosting_type    = params['bst_type'],
                               n_jobs           = 4)
        
        model.fit(x_train, y_train,
                  eval_set=[[x_train, y_train], [x_valid, y_valid]],
                  eval_metric='multi_logloss',
                  early_stopping_rounds=100,
                  verbose=100)

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
        save_model(model, f"./model/LGBM_model{cv_loss:.4f}.pkl")

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