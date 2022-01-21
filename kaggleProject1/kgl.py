# -*- coding: utf-8 -*-
"""
Created on Fri Jan 14 16:38:57 2022

@author: dipif
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt



class KGL :
    def __init__(self) :
        self.df = pd.read_csv('train.csv')
        np.random.seed(231)
    
    def get_basic_info(self) :
        print("Table")
        print(self.df)
        print("\n\nColumn information")
        print(self.df.info())
        print("\n\nStatistic properties")
        print(self.df.describe())
        print("\n\nNull counts")
        print(self.df.isnull().sum())
        print("\n\nGroup by classes")
        print(self.df.groupby(['country', 'store', 'product']).count())
        
    def show_basic_hist(self, bins_ = 50) :
        self.df.hist(bins = bins_)
        plt.show()
        
        
    def make_groups(self) :
        self.df_sold = pd.DataFrame()
        for i in self.df['country'].unique() :
            for j in self.df['store'].unique() :
                for k in self.df['product'].unique() :
                    init_idx_list = list(self.df[(self.df['country'] == i) & (self.df['store'] == j) & 
                                                        (self.df['product'] == k)]['num_sold'])
                    self.df_sold[k[7:] + ',' + j[6:] + ',' + i[0]] = pd.Series(init_idx_list, index = self.df['date'].unique())
        print(self.df_sold)
        
    def show_my_EDA(self) :
        plt.plot(self.df_sold.index, self.df_sold['Sticker,Mart,S'])
        plt.show()
        
        
    def temp_func(self) :
        print(self.df[(self.df['country'] == 'Sweden') & (self.df['product'] == 'Kaggle Sticker') & (self.df['store'] == 'KaggleMart')])
        
    def moving_average(self, day) :
        self.df_sold_ma = self.df_sold.rolling(window = 10).mean()
        plt.plot(self.df_sold.index, self.df_sold_ma['Sticker,Mart,S'])
        plt.show()
        
    def linear_regression1(self) :
        self.linear_coef = {}
        x = 1095 * 1096 / 2
        for i in self.df_sold.columns :
            temp_arr = self.df_sold[i].to_numpy()[:1096]
            y = temp_arr.sum()
            xy = (temp_arr * np.arange(1096)).sum()
            xx = (np.arange(1096, dtype = np.int64) * np.arange(1096, dtype = np.int64)).sum()
            a = (1096*xy - x * y)/(1096*xx - x*x)
            b = (y - a*x)/1096
            self.linear_coef[i] = [a, b]
            
    def show_linear_regression1(self) :
        x = np.arange(1096)
        y = self.df_sold['Sticker,Mart,S'][:1096]
        a, b = self.linear_coef['Sticker,Mart,S']
        y_linear = a*x + b
        plt.figure(figsize = (10, 10))
        plt.plot(x, y)
        line = plt.plot(x, y_linear)
        plt.setp(line, linewidth = 3.0)
        plt.show()
        
      
    # y는 2018년 1월 1일에서 2018년 12월 31일 까지 각 column별로 순서대로 담긴 ndarray
    def score(self, y) :
        score = 0
        cnt = 0
        for i in self.df_sold.columns :
            temp_y = self.df_sold[i].to_numpy()[1096:]
            score += ((temp_y - y[cnt])**2).mean()
            cnt += 1
        score /= len(self.df_sold.columns)
        print(score)
          
        
    def get_score1(self) :
        x = np.arange(1096, 1461)
        y = np.zeros(shape = (len(self.df_sold.columns), 365))
        cnt = 0
        for i in self.df_sold.columns :
            a = self.linear_coef[i][0]
            b = self.linear_coef[i][1]
            y[cnt] = a * x + b
            cnt += 1
        plt.figure(figsize = (10, 10))
        plt.plot(np.arange(365), self.df_sold[self.df_sold.columns[0]][1096:])
        plt.plot(np.arange(365), y[0])
        plt.show()
        self.score(y)
            
        
    #return의 a, b는 1년 단위의 세 점을 잇는 linear regression의 계수들, 길이 365의 ndarray
    def linear_regression2(self) :
        self.linear_coef2 = {}
        x = 3
        for i in self.df_sold.columns :
            y1 = self.df_sold[i].to_numpy()[:365]
            #2016년 윤년으로 인해 발생한 2월 29일은 생략
            y2 = np.concatenate((self.df_sold[i].to_numpy()[365:424], self.df_sold[i].to_numpy()[425:731]), axis = None)
            y3 = self.df_sold[i].to_numpy()[731:1096]
            y = y1 + y2 + y3
            xy = y2 + 2 * y3
            xx = 5
            a = (3 * xy - x * y)/(3*xx - x*x)
            b = (y - a*x)/3
            self.linear_coef2[i] = [a, b]
    
            
    def show_linear_regression2(self) :
        x = np.arange(1096)
        y = self.df_sold['Sticker,Mart,S'][:1096]
        y1 = self.linear_coef2['Sticker,Mart,S'][1]
        y2 = self.linear_coef2['Sticker,Mart,S'][0] + self.linear_coef2['Sticker,Mart,S'][1]
        y3 = self.linear_coef2['Sticker,Mart,S'][0] * 2 + self.linear_coef2['Sticker,Mart,S'][1]
        #윤년으로 사라진 2월 29일에 대한 예측을 2월 28일과 3월 1일의 평균으로 계산
        z = [(y2[58] + y2[59])/2]
        y2 = np.concatenate((y2[:59], np.array(z), y2[59:]), axis = None)
        y_linear = np.concatenate((y1, y2, y3), axis = None)
        plt.figure(figsize = (10, 10))
        plt.plot(x, y)
        plt.plot(x, y_linear)
        plt.show()
        
    
    def get_score2(self) :
        y = np.zeros(shape = (len(self.df_sold.columns), 365))
        cnt = 0
        for i in self.df_sold.columns :
            a = self.linear_coef2[i][0]
            b = self.linear_coef2[i][1]
            y[cnt] = a*3 + b
            cnt += 1
        plt.figure(figsize = (10, 10))
        plt.plot(np.arange(365), self.df_sold[self.df_sold.columns[0]][1096:])
        plt.plot(np.arange(365), y[0])
        plt.show()
        self.score(y)
        
    def percentage_regression(self) :
        for i in self.df_sold.columns :
            y1 = self.df_sold[i].to_numpy()[:365]
            y2 = np.concatenate((self.df_sold[i].to_numpy()[365:424], self.df_sold[i].to_numpy()[425:731]), axis = None)
            y3 = self.df_sold[i].to_numpy()[731:1096]
            y1 = np.log(y1)
            y2 = np.log(y2) - y1
            y3 = np.log(y3) - y1
            y1 -= y1
            self.percentage_coef = np.exp((y2.mean() + y3.mean())/2)
        
    def show_percentage_regression(self) :
        x = np.arange(1096)
        y = self.df_sold['Sticker,Mart,S'][:1096]
        y1 = self.df_sold['Sticker,Mart,S'][:365]
        y2 = self.percentage_coef*y1
        y3 = self.percentage_coef**2 * y1        
        z = [(y2[58] + y2[59])/2]
        y2 = np.concatenate((y2[:59], np.array(z), y2[59:]), axis = None)
        y_linear = np.concatenate((y1, y2, y3), axis = None)
        plt.figure(figsize = (10, 10))
        plt.plot(x, y)
        plt.plot(x, y_linear)
        plt.show()
        
    def get_percentage_score(self) :
        y = np.zeros(shape = (len(self.df_sold.columns), 365))
        cnt = 0
        for i in self.df_sold.columns :
            y1 = self.df_sold[i].to_numpy()[:365]
            y2 = np.concatenate((self.df_sold[i].to_numpy()[365:424], self.df_sold[i].to_numpy()[425:731]), axis = None)
            y3 = self.df_sold[i].to_numpy()[731:1096]
            y[cnt] = np.power(y1, 1/6) * np.power(y2, 1/3) * np.power(y3, 1/2) * self.percentage_coef**2
            cnt += 1
        plt.figure(figsize = (10, 10))
        plt.plot(np.arange(365), self.df_sold[self.df_sold.columns[0]][1096:])
        plt.plot(np.arange(365), y[0])
        plt.show()
        self.score(y)
        
    def get_percentage_score2(self) :
        y = np.zeros(shape = (len(self.df_sold.columns), 365))
        cnt = 0
        for i in self.df_sold.columns :
            y3 = self.df_sold[i].to_numpy()[731:1096]
            y[cnt] = y3 * self.percentage_coef
            cnt += 1
        self.score(y)
        
    def get_score_default(self) :
        y = np.zeros(shape = (len(self.df_sold.columns), 365))
        cnt = 0
        for i in self.df_sold.columns :
            y[cnt] = self.df_sold[i].to_numpy()[731:1096]
            cnt+=1
        self.score(y)
        
    
    def answer_regression(self) :
        test_df = pd.read_csv('test.csv')
        test_df.insert(1, 'num_sold', 0.)
        for i in self.df['country'].unique() :
            for j in self.df['store'].unique() :
                for k in self.df['product'].unique() :
                    self.var = self.df[(self.df['country'] == i) & (self.df['store'] == j) & 
                                                        (self.df['product'] == k)]['num_sold'].to_numpy()
                    print(self.var)
                    y2 = np.concatenate((self.var[365:424], self.var[425:731]), axis = None)
                    y3 = self.var[731:1096]
                    y4 = self.var[1096:]
                    y2 = np.log(y2)
                    y3 = np.log(y3) - y2
                    y4 = np.log(y4) - y2
                    y2 -= y2
                    coef= np.exp((y3.mean() + y4.mean())/3)
                    y2 = np.concatenate((self.var[365:424], self.var[425:731]), axis = None)
                    y3 = self.var[731:1096]
                    y4 = self.var[1096:]
                    y = np.power(y2, 1/4) * np.power(y3, 1/2) * np.power(y4, 1/4) * coef**2
                    test_df.loc[(test_df['country'] == i) & (test_df['store'] == j) & (test_df['product'] == k), ['num_sold']] = y

        test_df.set_index(np.arange(26298, 32868), inplace = True)
        test_df.index.name = 'row_id'
        print(test_df)
        test_df['num_sold'].to_csv('aa.csv')
                            
                    
  