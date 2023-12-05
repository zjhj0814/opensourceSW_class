import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

def sort_dataset(dataset_df):
    return dataset_df.sort_values(by='year')

def split_dataset(dataset_df):
    Y = dataset_df[['salary']]
    Y = Y.mul(0.001)
    X = dataset_df.drop(columns="salary", axis=1)

    split_index = 1718  # 훈련 및 테스트 데이터셋을 나누는 인덱스

    X_train, X_test = X.iloc[:split_index], X.iloc[split_index:]
    Y_train, Y_test = Y.iloc[:split_index], Y.iloc[split_index:]
    return X_train, X_test, Y_train, Y_test

def extract_numerical_cols(dataset_df):
    return dataset_df[['age', 'G', 'PA', 'AB', 'R', 'H', '2B', '3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'HBP', 'SO', 'GDP', 'fly', 'war']]

def train_predict_decision_tree(X_train, Y_train, X_test):
    dt_reg = DecisionTreeRegressor()
	#classifier는 분류문제를 다루기 위한 모델, regressor는 회귀 문제를 다루기 위한 모델.
	#연속적인 값을 가진 salary를 예측하는 문제이므로 회귀 모델을 사용하는 게 좋다.
    dt_reg.fit(X_train, Y_train.values.ravel()) #ravel로 2D를 1D 배열로 변환.
    return dt_reg.predict(X_test)

def train_predict_random_forest(X_train, Y_train, X_test):
    rf_reg = RandomForestRegressor()
    rf_reg.fit(X_train, Y_train.values.ravel())
    return rf_reg.predict(X_test)

def train_predict_svm(X_train, Y_train, X_test):
    svm_pipe = make_pipeline(
        StandardScaler(),
        SVR()
    )
    svm_pipe.fit(X_train, Y_train.values.ravel())
    return svm_pipe.predict(X_test)

def calculate_RMSE(labels, predictions):
    return np.sqrt(mean_squared_error(labels, predictions))

if __name__ == '__main__':
    data_df = pd.read_csv("C:\\Users\\82104\\OneDrive\\문서\\오픈소스 SW개론\\과제2\\2019_kbo_for_kaggle_v2.csv")

    sorted_df = sort_dataset(data_df)
    X_train, X_test, Y_train, Y_test = split_dataset(sorted_df)

    X_train = extract_numerical_cols(X_train)
    X_test = extract_numerical_cols(X_test)

    dt_predictions = train_predict_decision_tree(X_train, Y_train, X_test)
    rf_predictions = train_predict_random_forest(X_train, Y_train, X_test)
    svm_predictions = train_predict_svm(X_train, Y_train, X_test)

    print("Decision Tree Test RMSE: ", calculate_RMSE(Y_test, dt_predictions))
    print("Random Forest Test RMSE: ", calculate_RMSE(Y_test, rf_predictions))
    print("SVM Test RMSE: ", calculate_RMSE(Y_test, svm_predictions))
