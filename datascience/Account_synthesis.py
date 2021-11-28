"""
Module dedicated to prediction of next monthly outgoing of a given account, based on last 6 months of history
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import linear_model as lin_mdl
import json


class Account_synthesis:
    """
    Account_synthesis class
      + Properties
        - history: synthesis of account history (date, transaction amount, balance, cumulated outgoing, cumulated income)
        - history_length: history length [number of days]
        - update_date: update date
        - update_balance: balance @update date
        - analysis_date: analysis date
        - history_length_at_analysis_date: history length @analysis date [number of days]
        - last_6_month: summarized data of last 6 months @analysis date (date, cumulated income, cumulated outgoing, balance, monthly income, monthly outgoing)
        - next_monthly_outgoing: Next monthly outgoing, if available (>30 days of history after analysis date) (useful for evaluation of prediction model accuracy)
      + Methods
        - get_last_6_month(analysis_date = None): get last 6 month analysis at analysis date
        - plot_history(): Synthesis plot of account history
        - plot_last_6_month(): Synthesis plot of last 6 months (at analysis date)
        - monthly_outgoing_to_predict(plot_res=False):Computes next monthly outgoing to predict, depending on chosen analysis date ; return vector of analysis date and associated next monthly outgoing. Plot it if plot_res=True
        - predict_next_monthly_outgoing(mdl_type = "Linear model based on studied account ; Extrapolate outgoing"): next monthly prediction
            3 models available :
              - mdl_type = "Linear model based on studied account ; Extrapolate outgoing"
              - mdl_type = "Linear model based on studied account ; Extrapolate income and ensure stable balance"
              - mdl_type = "Linear model based on large sample of accounts"
    """

    def __init__(self,account, transactions, analysis_date = None):
        """
        account: DataFrame with 1 row and 2 columns : update_date, balance
            => Gives account balance at update date
        transactions: DateFrame with n rows and 2 columns : date, amount
            => Gives transactions history on this account
        """
        transactions = transactions.sort_values(by="date", ascending=False)
        nb_trans = len(transactions.index)  # Number of transactions

        # Initialization of account history
        account_history = pd.DataFrame(data={'date': [account.update_date.iloc[0]],
                                             'amount': [0],
                                             'balance': [account.balance.iloc[0]],
                                             'outgoing_cumsum': [0],
                                             'income_cumsum': [0]})
        if nb_trans > 0:
          account_history = pd.concat([account_history, transactions], ignore_index=True)

          # Calculating balance column
          account_history.loc[1:nb_trans, 'balance'] = account_history.balance.iloc[0] - pd.Series(
                                                        np.array(account_history.amount.iloc[0:nb_trans - 1].cumsum()),
                                                        account_history.iloc[1:nb_trans].index)  # balance history

          # Calculating cumulated outgoing column
          account_history.loc[:, 'outgoing_cumsum'] = 0  # initialization
          account_history.outgoing_cumsum = pd.Series(np.array(account_history.amount.iloc[::-1].apply(lambda x: -min(0, x)).cumsum()),
                                                      account_history.iloc[::-1].index)  # cumulated outgoing

          # Calculating cumulated income
          account_history.loc[:, 'income_cumsum'] = 0  # initialization
          account_history.income_cumsum = pd.Series(np.array(account_history.amount.iloc[::-1].apply(lambda x: max(0, x)).cumsum()),
                                                             account_history.iloc[::-1].index)  # cumulated income

        self.history = account_history #synthesis of account history (date, transaction amount, balance, cumulated outgoing, cumulated income)
        self.history_length = (self.history.date.max()-self.history.date.min()).days #history length [number of days]
        self.update_date = self.history.date.max()  # update date
        self.update_balance = self.history.balance.iloc[0]  # balance @update date
        self.get_last_6_month(analysis_date) # summarized data of last 6 months (date, cumulated income, cumulated outgoing, balance, monthly income, monthly outgoing)

    def get_last_6_month(self, analysis_date = None):
        '''
        Returns the condensed data of the last 6 months from date "analysis_date" (= update date by default)
        (date, cumulated income, cumulated outgoing, balance, monthly income, monthly outgoing, next monthly income, next monthly outgoing)
        '''
        #date used for last 6 month analysis
        if analysis_date != None:
            self.analysis_date=np.min([analysis_date,self.update_date]) #specified by user, but can't be more recent than update date
            self.analysis_date=pd.DatetimeIndex([self.analysis_date])[0] #Convert to TimeStamp (quite strange !!)
        else:
            self.analysis_date = self.update_date #default : account update date
        
        self.history_length_at_analysis_date = np.max([0,(self.analysis_date-self.history.date.min()).days]) #history length at analysis date [number of days]
        
        
        if (self.history_length_at_analysis_date >= (6 * 30)):
            #Resampling by 30 days periods (with analysis date as origin)
            monthly_data = self.history.set_index("date").resample("30D", origin=self.analysis_date, label="right", closed="right")["income_cumsum", "outgoing_cumsum", "balance"].agg({"income_cumsum": "max", "outgoing_cumsum": "max", "balance": "mean"})
            #Interpolation of missing values
            monthly_data.interpolate(inplace=True) 
            #Calculate monthly income & outgoing
            monthly_data["monthly_income"] = monthly_data.income_cumsum.diff()
            monthly_data["monthly_outgoing"] = monthly_data.outgoing_cumsum.diff()
            #Selecting last 6 months from analysis date
            last_6_month = monthly_data[(monthly_data.index<=self.analysis_date) & (monthly_data.index>(self.analysis_date-np.timedelta64(6*30,'D')))]
            #Next monthly outgoing, if available (useful for evaluation of prediction model accuracy)
            if np.any(monthly_data.index==(self.analysis_date+np.timedelta64(30,'D'))):
                #Month M+1 found
                next_monthly_outgoing = monthly_data.monthly_outgoing.loc[monthly_data.index==(self.analysis_date+np.timedelta64(30,'D'))]
            else:
                #Analysis date is in last 30 days of history => no available next monthly outgoing
                next_monthly_outgoing = None
        else:
            last_6_month = pd.DataFrame(columns={'date', 'income_cumsum', 'outgoing_cumsum', 'balance', 'monthly_income',
                                                 'monthly_outgoing'}).set_index('date')
            next_monthly_outgoing = None

        self.last_6_month=last_6_month
        self.next_monthly_outgoing=next_monthly_outgoing
        

    def plot_history(self):
        """
        Synthesis plot of account history
        """
        plt.figure(figsize=(12, 6))
        ax1 = plt.axes()
        ax1.plot(self.history.date, self.history.balance, 'b+-')
        ax1.set_xlabel("Date"), ax1.set_ylabel("Balance [€]", color="b")
        ax1.tick_params(axis='y', color='b', labelcolor='b')

        ax2 = ax1.twinx()
        ax2.plot(self.history.date, self.history.outgoing_cumsum, 'r+-', label="Cumulated outgoing")
        ax2.plot(self.history.date, self.history.income_cumsum, 'g+-', label="Cumulated income")
        ax2.set_ylabel("Cumulated income / outgoing [€]")
        ax2.legend()
        ax2.spines["left"].set_color("b")

        plt.title("Account: {:.2f}€ @{} - {:d} days of history".format(self.update_balance,
                                                                       self.update_date.strftime('%Y-%m-%d'),
                                                                       self.history_length))
        plt.show()
        
    def plot_last_6_month(self):
        """
        Synthesis plot of last 6 months
        """
        if self.last_6_month.size > 0 :
            plt.figure(figsize=(18,12))
            plt.subplot(221)
            plt.plot(self.history.date,self.history.balance,'b',label="Balance")
            plt.plot(self.last_6_month.index,self.last_6_month.balance,'c*-',linewidth=4,label="Balance (mean resample @last 6 month)")
            plt.legend()
            plt.ylabel("Balance [€]")
            
            plt.subplot(223)
            plt.plot(self.history.date,self.history.outgoing_cumsum,'r',label="Cumulated outgoing")
            plt.plot(self.last_6_month.index,self.last_6_month.outgoing_cumsum,'mo-',linewidth=4,label="Cumulated outgoing (resample @last 6 month)")
            plt.plot(self.history.date,self.history.income_cumsum,'g',label="Cumulated income")
            plt.plot(self.last_6_month.index,self.last_6_month.income_cumsum,'+-',color="lightgreen",linewidth=4,label="Cumulated income (resample @last 6 month)")
            plt.ylabel("[€]")
            plt.legend()
            
            plt.subplot(122)
            plt.plot(self.last_6_month.index,self.last_6_month.monthly_income,'o-',color="lightgreen",label="Monthly income")
            plt.plot(self.last_6_month.index,self.last_6_month.monthly_outgoing,'mo-',label="Monthly outgoing")
            if not(self.next_monthly_outgoing is None):
                plt.plot(self.next_monthly_outgoing.index,self.next_monthly_outgoing,'m*',label="Next monthly outgoing to predict")
            plt.ylabel("[€]")
            plt.legend()
            
            plt.show()
        else :
            print("No sufficient history")

    def monthly_outgoing_to_predict(self,plot_res=False):
        """
        Computes next monthly outgoing to predict, depending on chosen analysis date
        """
        if self.history_length >= (7*30): #7 months history required (6 month required for prediction + 1 month for next monthly outgoing calculation)
            #Creation analysis date range
            min_date=self.history.date.min()+np.timedelta64(6*30,'D')
            max_date=self.history.date.max()-np.timedelta64(30,'D')
            X_analysis_date=pd.date_range(min_date,max_date,freq='D')
            
            analysis_date_saved = self.analysis_date #save analysis date to restore it after testing different analysis date
            next_monthly_outgoing = pd.Series([],name="monthly_outgoing",dtype="float64") #initialization
            for date_ii in X_analysis_date:
                self.get_last_6_month(date_ii)
                if self.last_6_month.size == 0 :
                    print("ko")
                    print(date_ii)
                next_monthly_outgoing=next_monthly_outgoing.append(self.next_monthly_outgoing)
                
            Y_next_monthly_outgoing=next_monthly_outgoing.values
            
            self.get_last_6_month(analysis_date_saved) #Restore initial analysis date
            
            if plot_res == True:
                plt.figure(figsize=(12,6))
                plt.plot(X_analysis_date,Y_next_monthly_outgoing,'m-',label="To predict")
                plt.xlabel("Analysis date")
                plt.ylabel("Next monthly outgoing [€]")
                plt.legend()
            
            return X_analysis_date, Y_next_monthly_outgoing
            
        else:
            print("No sufficient history for validation of prediction model")
            return None, None
        
        
    def predict_next_monthly_outgoing(self, mdl_type = "Linear model based on studied account ; Extrapolate outgoing"):
        '''
        Next monthly prediction
        3 models available :
          - mdl_type = "Linear model based on studied account ; Extrapolate outgoing"
          - mdl_type = "Linear model based on studied account ; Extrapolate income and ensure stable balance"
          - mdl_type = "Linear model based on large sample of accounts"
        '''
        
        if (self.history_length_at_analysis_date >= (6 * 30)):
            if mdl_type == "Linear model based on studied account ; Extrapolate outgoing":
                #Linear model
                mdl=lin_mdl.LinearRegression()
                #Train set (last 6 months outgoing)
                X_train=np.arange(1,7).reshape(-1,1) #6 months
                Y_train=self.last_6_month.monthly_outgoing.values.reshape(-1,1) #Associated monthly outgoing
                #Fit
                mdl.fit(X_train,Y_train)
                #Predict
                X_test=[[7]] #Next month
                Y_predict=np.maximum(0,mdl.predict(X_test)) #Ensure >=0 prediction
                next_monthly_outgoing = Y_predict[0,0] #Scalar value
                
            elif mdl_type == "Linear model based on studied account ; Extrapolate income and ensure stable balance":
                #Linear model for income
                mdl=lin_mdl.LinearRegression()
                #Train set (last 6 months)
                X_train=np.arange(1,7).reshape(-1,1) #6 months
                Y_train_income=self.last_6_month.monthly_income.values.reshape(-1,1) #Associated monthly income
                Y_train_outgoing=self.last_6_month.monthly_outgoing.values.reshape(-1,1) #Associated monthly outgoing
                #Fit income model
                mdl.fit(X_train,Y_train_income)
                #Predict
                X_test=[[7]] #Next month
                Y_predict_income = np.maximum(0,mdl.predict(X_test)) #Predict next monthly income, and ensure >=0 prediction
                Y_predict_outgoing = np.maximum(0,np.sum(Y_train_income)+Y_predict_income[0,0]-np.sum(Y_train_outgoing)) #Predict next monthly outgoing
                next_monthly_outgoing = Y_predict_outgoing
                
            elif mdl_type == "Linear model based on large sample of accounts":
                #Read model parameters from saved text file (JSON format)
                f=open('mdl_3_param.txt', 'r')
                mdl_param = json.load(f)
                f.close()
                #Create model and set parameters
                mdl=lin_mdl.LinearRegression()
                mdl.coef_ = np.array(mdl_param['coef'])
                mdl.intercept_ = np.array(mdl_param['intercept'])
                #Predict
                X_test=np.hstack((self.last_6_month.monthly_outgoing.values,self.last_6_month.monthly_income.values)).reshape(1, -1) #data
                Y_predict=np.maximum(0,mdl.predict(X_test)) #Ensure >=0 prediction
                next_monthly_outgoing = Y_predict[0] #Scalar value
                
            else:
                print("Unknown model !")
                next_monthly_outgoing = None
        else:
            print("No sufficient history")
            next_monthly_outgoing = None
            
        return next_monthly_outgoing

    def __str__(self):
        """
        String representation of current Account_synthesis instance
        """
        return "Account synthesis:\n  @ Update date ({}): {:.2f}€ - {:d} days of history\n  @ Analysis date ({}): {:d} days of history".format(self.update_date.strftime('%Y-%m-%d'),self.update_balance,self.history_length,
                                                                                                                                             self.analysis_date.strftime('%Y-%m-%d'),self.history_length_at_analysis_date)
