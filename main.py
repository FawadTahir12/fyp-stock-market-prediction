#**************** IMPORT PACKAGES ********************
from flask import Flask, render_template, request, flash, redirect, url_for
from alpha_vantage.timeseries import TimeSeries
import pandas as pd
import numpy as np
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error, accuracy_score
import matplotlib.pyplot as plt
plt.style.use('ggplot')
import math, random
from datetime import datetime
import datetime as dt
import yfinance as yf
import tweepy
import preprocessor as p
import re
from sklearn.linear_model import LinearRegression
from textblob import TextBlob
import constants as ct
from Tweet import Tweet
import nltk

nltk.download('punkt')

# Ignore Warnings
import warnings
warnings.filterwarnings("ignore")
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

#***************** FLASK *****************************
app = Flask(__name__)


@app.after_request
def add_header(response):
    response.headers['Pragma'] = 'no-cache'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Expires'] = '0'
    return response

@app.route('/')
def index():
   return render_template('index.html')

@app.route('/prediction',methods = ['POST'])
def prediction():
    nm = request.form['nm']

    #**************** FUNCTIONS TO FETCH DATA ***************************
    def get_historical(quote):
        end = datetime.now()
        start = datetime(end.year-2,end.month,end.day)
        data = yf.download(quote, start=start, end=end)  #get_data(qu, start_date = start, end_date = end, index_as_date = True, interval = “1d”)
        df = pd.DataFrame(data=data)
        df.to_csv(''+quote+'.csv')
        if(df.empty):
            ts = TimeSeries(key='N6A6QT6IBFJOPJ70',output_format='pandas')
            data, meta_data = ts.get_daily_adjusted(symbol='NSE:'+quote, outputsize='full')
            data=data.head(503).iloc[::-1]
            data=data.reset_index()
            #Keep Required cols only
            df=pd.DataFrame()
            df['Date']=data['date']
            df['Open']=data['1. open']
            df['High']=data['2. high']
            df['Low']=data['3. low']
            df['Close']=data['4. close']
            df['Adj Close']=data['5. adjusted close']
            df['Volume']=data['6. volume']
            df.to_csv(''+quote+'.csv',index=False)
        return

    #******************** ARIMA SECTION ********************
    def ARIMA_ALGO(df):
        uniqueVals = df["Code"].unique() 
        print(uniqueVals) 
        len(uniqueVals)
        df=df.set_index("Code")
        
        def parser(x):
            return datetime.strptime(x, '%Y-%m-%d')
        def arima_model(train, test):
            history = [x for x in train]
            predictions = list()
            for t in range(len(test)):
                model = ARIMA(history, order=(6,1 ,0))
                model_fit = model.fit(disp=0)
                output = model_fit.forecast()
                yhat = output[0]
                predictions.append(yhat[0])
                obs = test[t]
                history.append(obs)
            return predictions
        for company in uniqueVals[:10]:
            data=(df.loc[company,:]).reset_index()
            data['Price'] = data['Close']
            Quantity_date = data[['Price','Date']]
            Quantity_date.index = Quantity_date['Date'].map(lambda x: parser(x))
            Quantity_date['Price'] = Quantity_date['Price'].map(lambda x: float(x))
            Quantity_date = Quantity_date.fillna(Quantity_date.bfill())
            Quantity_date = Quantity_date.drop(['Date'],axis =1)
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(Quantity_date)
            plt.savefig('static/Trends.png')
            plt.close(fig)
            
            quantity = Quantity_date.values
            size = int(len(quantity) * 0.80)
            train, test = quantity[0:size], quantity[size:len(quantity)]
            #fit in model
            predictions = arima_model(train, test)
            forecast_list = predictions[-7:]
            print(forecast_list)
            #plot graph
            fig = plt.figure(figsize=(7.2,4.8),dpi=65)
            plt.plot(test,label='Actual Price')
            plt.plot(predictions,label='Predicted Price')
            plt.legend(loc=4)
            plt.savefig('static/ARIMA.png')
            plt.close(fig)
            print()
            print("##############################################################################")
            arima_pred=predictions[-2]
            print("Tomorrow's",quote," Closing Price Prediction by ARIMA:",arima_pred)
            #rmse calculation
            error_arima = math.sqrt(mean_squared_error(test, predictions))
            # accuracy = round(accuracy_score(test, predictions)*100, 2)
            # print(accuracy)
            print("ARIMA RMSE:",error_arima)
            print("##############################################################################")



            return arima_pred, error_arima, forecast_list
        
        


    #************* LSTM SECTION **********************

    def LSTM_ALGO(df):
        dataset_train=df.iloc[0:int(0.8*len(df)),:]
        dataset_test=df.iloc[int(0.8*len(df)):,:]
        training_set=df.iloc[:,4:5].values
        from sklearn.preprocessing import MinMaxScaler
        sc=MinMaxScaler(feature_range=(0,1))
        training_set_scaled=sc.fit_transform(training_set)
        X_train=[]
        y_train=[]
        for i in range(7,len(training_set_scaled)):
            X_train.append(training_set_scaled[i-7:i,0])
            y_train.append(training_set_scaled[i,0])
        X_train=np.array(X_train)
        
        y_train=np.array(y_train)
        X_forecast=np.array(X_train[-1,1:])
        X_forecast=np.append(X_forecast,y_train[-1])
        X_train=np.reshape(X_train, (X_train.shape[0],X_train.shape[1],1))
        X_forecast=np.reshape(X_forecast, (1,X_forecast.shape[0],1))
        from keras.models import Sequential
        from keras.layers import Dense
        from keras.layers import Dropout
        from keras.layers import LSTM
    
        regressor=Sequential()
        regressor.add(LSTM(units=50,return_sequences=True,input_shape=(X_train.shape[1],1)))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50,return_sequences=True))
        regressor.add(Dropout(0.1))
        regressor.add(LSTM(units=50))
        regressor.add(Dropout(0.1))
        regressor.add(Dense(units=1))
        
        regressor.compile(optimizer='adam',loss='mean_squared_error')
        regressor.fit(X_train,y_train,epochs=30,batch_size=32 )
        real_stock_price=dataset_test.iloc[:,4:5].values
        dataset_total=pd.concat((dataset_train['Close'],dataset_test['Close']),axis=0) 
        testing_set=dataset_total[ len(dataset_total) -len(dataset_test) -7: ].values
        testing_set=testing_set.reshape(-1,1)
        testing_set=sc.transform(testing_set)
        X_test=[]
        for i in range(7,len(testing_set)):
            X_test.append(testing_set[i-7:i,0])
        X_test=np.array(X_test)
        X_test=np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
        predicted_stock_price=regressor.predict(X_test)
        predicted_stock_price=sc.inverse_transform(predicted_stock_price)
        # print(predicted_stock_price)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        plt.plot(real_stock_price,label='Actual Price')  
        plt.plot(predicted_stock_price,label='Predicted Price')
          
        plt.legend(loc=4)
        plt.savefig('static/LSTM.png')
        plt.close(fig)     
        error_lstm = math.sqrt(mean_squared_error(real_stock_price, predicted_stock_price))
        # accuracy = accuracy_score(real_stock_price, predicted_stock_price)
        # print(accuracy)
        forecasted_stock_price=regressor.predict(X_forecast)
        forecasted_stock_price=sc.inverse_transform(forecasted_stock_price)
        # print("$$$$$$$$$$$$$$$ Forecasted LSTM",forecasted_stock_price)
        
        lstm_pred=forecasted_stock_price[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by LSTM: ",lstm_pred)
        print("LSTM RMSE:",error_lstm)
        print("##############################################################################")

        return lstm_pred,error_lstm
    #***************** LINEAR REGRESSION SECTION ******************       
    def LIN_REG_ALGO(df):
        forecast_out = int(7)
        df['Close after n days'] = df['Close'].shift(-forecast_out)
        df_new=df[['Close','Close after n days']]
        y =np.array(df_new.iloc[:-forecast_out,-1])
        y=np.reshape(y, (-1,1))
        X=np.array(df_new.iloc[:-forecast_out,0:-1])
        X_to_be_forecasted=np.array(df_new.iloc[-forecast_out:,0:-1])

        X_train=X[0:int(0.8*len(df)),:]
        X_test=X[int(0.8*len(df)):,:]
        y_train=y[0:int(0.8*len(df)),:]
        y_test=y[int(0.8*len(df)):,:]

        from sklearn.preprocessing import StandardScaler
        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)        
        X_to_be_forecasted=sc.transform(X_to_be_forecasted)
        clf = LinearRegression(n_jobs=-1)

        clf.fit(X_train, y_train)
        y_test_pred=clf.predict(X_test)
        y_test_pred=y_test_pred*(1.04)

        
        import matplotlib.pyplot as plt2
        fig = plt2.figure(figsize=(7.2,4.8),dpi=65)
        plt2.plot(y_test,label='Actual Price' )
        plt2.plot(y_test_pred,label='Predicted Price')
        
        plt2.legend(loc=4)
        plt2.savefig('static/LR.png')
        plt2.close(fig)
        error_lr = math.sqrt(mean_squared_error(y_test, y_test_pred))
        
        
        forecast_set = clf.predict(X_to_be_forecasted)
        forecast_set=forecast_set*(1.04)
        mean=forecast_set.mean()
        lr_pred=forecast_set[0,0]
        print()
        print("##############################################################################")
        print("Tomorrow's ",quote," Closing Price Prediction by Linear Regression: ",lr_pred)
        print("Linear Regression RMSE:",error_lr)
        print("##############################################################################")


        return df, lr_pred, forecast_set, mean, error_lr
    #**************** SENTIMENT ANALYSIS **************************
    def retrieving_tweets_polarity(symbol):
        stock_ticker_map = pd.read_csv('Yahoo-Finance-Ticker-Symbols.csv')
        stock_full_form = stock_ticker_map[stock_ticker_map['Ticker']==symbol]
        symbol = stock_full_form['Name'].to_list()[0][0:12]
        auth = tweepy.OAuthHandler(ct.consumer_key, ct.consumer_secret)
        auth.set_access_token(ct.access_token, ct.access_token_secret)
        user = tweepy.API(auth)        
        tweets = tweepy.Cursor(user.search_tweets, q=symbol, tweet_mode='extended', lang='en',exclude_replies=True).items(ct.num_of_tweets)       
        tweet_list = [] 
        global_polarity = 0 
        tw_list=[] 
        pos=0 
        neg=1 
        for tweet in tweets:
            count=20 
            tw2 = tweet.full_text
            tw = tweet.full_text
            tw=p.clean(tw)
            tw=re.sub('&amp;','&',tw)
            tw=re.sub(':','',tw)
            tw=tw.encode('ascii', 'ignore').decode('ascii')
            blob = TextBlob(tw)
            polarity = 0 
            for sentence in blob.sentences:            
                polarity += sentence.sentiment.polarity
                if polarity>0:
                    pos=pos+1
                if polarity<0:
                    neg=neg+1               
                global_polarity += sentence.sentiment.polarity
            if count > 0:
                tw_list.append(tw2)               
            tweet_list.append(Tweet(tw, polarity))
            count=count-1
        if len(tweet_list) != 0:
            global_polarity = global_polarity / len(tweet_list)
        else:
            global_polarity = global_polarity
        neutral=ct.num_of_tweets-pos-neg
        if neutral<0:
            neg=neg+neutral
            
        print()
        print("##############################################################################")
        print("Positive Tweets :",pos,"Negative Tweets :",neg,"Neutral Tweets :",neutral)
        print("##############################################################################")
        labels=['Positive','Negative','Neutral']
        sizes = [pos, neg, neutral]
        print(sizes)
        explode = (0, 0, 0)
        fig = plt.figure(figsize=(7.2,4.8),dpi=65)
        fig1, ax1 = plt.subplots(figsize=(7.2,4.8),dpi=65)
        ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%', startangle=90)
        ax1.axis('equal')  
        plt.tight_layout()
        plt.savefig('static/SA.png')
        plt.close(fig)
        if global_polarity>0:
            print()
            print("##############################################################################")
            print("Tweets Polarity: Overall Positive")
            print("##############################################################################")
            tw_pol="Overall Positive"
        else:
            print()
            print("##############################################################################")
            print("Tweets Polarity: Overall Negative")
            print("##############################################################################")
            tw_pol="Overall Negative"


        return global_polarity,tw_list,tw_pol,pos,neg,neutral

    def new_func(pos, neg, neutral):
        sizes = [pos,neg,neutral]
        return sizes


    def recommending(df, global_polarity,today_stock,mean):
        if today_stock.iloc[-1]['Close'] < mean:
            if global_polarity > 0:
                idea="RISE"
                decision="BUY"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
            elif global_polarity <= 0:
                idea="FALL"
                decision="SELL"
                print()
                print("##############################################################################")
                print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        else:
            idea="FALL"
            decision="SELL"
            print()
            print("##############################################################################")
            print("According to the ML Predictions and Sentiment Analysis of Tweets, a",idea,"in",quote,"stock is expected => ",decision)
        return idea, decision





    #**************GET DATA ***************************************
    quote=nm
    #Try-except to check if valid stock symbol
    try:
        get_historical(quote)
    except:
        return render_template('index.html',not_found=True)
    else:
    
        #************** PREPROCESSUNG ***********************
        df = pd.read_csv(''+quote+'.csv')
        print("##############################################################################")
        print("Today's",quote,"Stock Data: ")
        today_stock=df.iloc[-1:]
        print(today_stock)
        print("##############################################################################")
        df = df.dropna()
        code_list=[]
        for i in range(0,len(df)):
            code_list.append(quote)
        df2=pd.DataFrame(code_list,columns=['Code'])
        df2 = pd.concat([df2, df], axis=1)
        df=df2
        
        # arima_results = []
        # lstm_results = []
        # linear_results = []
        # tweets_results = []

        # t1 = Thread(target=ARIMA_ALGO, args=(df,arima_results))
        # t2 = Thread(target=LSTM_ALGO, args=(df,lstm_results))
        # t3 = Thread(target=LIN_REG_ALGO, args=(df,linear_results))
        # t4 = Thread(target=retrieving_tweets_polarity, args=(quote,tweets_results))

        # t1.start()
        # t2.start()
        # t3.start()
        # t4.start()

        # t1.join()
        # t2.join()
        # t3.join()
        # t4.join()

        # arima_pred, error_arima=arima_results
        # lstm_pred, error_lstm=lstm_results
        # df, lr_pred, forecast_set,mean,error_lr=linear_results
        # polarity,tw_list,tw_pol,pos,neg,neutral = tweets_results

        arima_pred, error_arima, forecast_list=ARIMA_ALGO(df)
        lstm_pred, error_lstm=LSTM_ALGO(df)
        df, lr_pred, forecast_set,mean,error_lr=LIN_REG_ALGO(df)
        polarity,tw_list,tw_pol,pos,neg,neutral = retrieving_tweets_polarity(quote)
        
        idea, decision=recommending(df, polarity,today_stock,mean)
        print()
        print("Forecasted Prices for Next 7 days:")
        print(forecast_list)
        today_stock=today_stock.round(2)
        return render_template('results.html',quote=quote,arima_pred=round(arima_pred,2),forecast_list=forecast_list, lstm_pred=round(lstm_pred,2),
                               lr_pred=round(lr_pred,2),open_s=today_stock['Open'].to_string(index=False),
                               close_s=today_stock['Close'].to_string(index=False),adj_close=today_stock['Adj Close'].to_string(index=False),
                               tw_list=tw_list,tw_pol=tw_pol,idea=idea,decision=decision,high_s=today_stock['High'].to_string(index=False),
                               low_s=today_stock['Low'].to_string(index=False),vol=today_stock['Volume'].to_string(index=False),
                               error_lr=round(error_lr,2),error_lstm=round(error_lstm,2),error_arima=round(error_arima,2))
if __name__ == '__main__':
   app.run()
   

















