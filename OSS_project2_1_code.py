import pandas as pd
import pandas_datareader.data as web

def problem1(data, year):
    data_year = data.loc[data['year'] == year, ['batter_name', 'H', 'avg', 'HR', 'OBP']]
    data_H = data_year.sort_values(by='H', ascending=False)
    data_avg = data_year.sort_values(by='avg', ascending=False)
    data_HR = data_year.sort_values(by='HR', ascending=False)
    data_OBP = data_year.sort_values(by='OBP', ascending=False)
    print(f"---{year}---")
    print("---top 10 players in hits---")
    print(data_H.head(10))
    print("---top 10 players in batting average---")
    print(data_avg.head(10))
    print("---top 10 players in homerun---")
    print(data_HR.head(10))
    print("---top 10 players in on base percentage---")
    print(data_OBP.head(10))

def problem2(data, position):
    data_2018_cp = data.loc[(data['year'] == 2018) & (data['cp'] == position), ['batter_name', 'cp', 'war']]
    data_position = data_2018_cp.sort_values(by='war', ascending=False)
    print(f"---player with the highest war by {position}---")
    print(data_position.head(1))

def problem3(data):
    data = data[['R', 'H', 'HR', 'RBI', 'SB', 'war', 'avg', 'OBP', 'SLG', 'salary']]
    data_corr = data.corrwith(data.salary) #series 리턴
    data_corr = data_corr.sort_values(ascending=False)
    print(data_corr.index[1]) #salary가 1번째 행일테니까, salary 다음으로 corr이 높은 index를 출력한다.

def main():
    filename = 'C:\\Users\\82104\\OneDrive\\문서\\오픈소스 SW개론\\과제2\\2019_kbo_for_kaggle_v2.csv'
    data = pd.read_csv(filename)
    print("\n---problem1---")
    for i in range(2015, 2018):
        problem1(data, i)
    print("\n---problem2---")
    position_info = ["포수", "1루수", "2루수", "3루수", "유격수", "좌익수", "중견수", "우익수"]
    for position in position_info:
        problem2(data, position)
    print("\n---problem3---")
    problem3(data)

if __name__=="__main__":
    main()
