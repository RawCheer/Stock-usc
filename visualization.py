#!/usr/bin/env python
# coding: utf-8

# Please download this file and execute it on your local computer
# 
# **references**
# 
# https://www.statworx.com/at/blog/how-to-build-a-dashboard-in-python-plotly-dash-step-by-step-tutorial/
# https://github.com/STATWORX/blog/blob/master/DashApp/app_basic.py
# 
# https://pierpaolo28.github.io/blog/blog21/
# https://github.com/pierpaolo28/Data-Visualization/tree/master/Dash

# Sentiment analysis: not live prediction
# 
# Duration: 2867sec = 48 min for twitter of all stocks
# 
# news: only AAPL, MSFT
# Duration: 381sec = 6 min for news of one stock
# 
# 
# XLK, QQQ, GOOG, BOTZ, NFLX, AMZN, FB ApiException: (404) Reason: Not Found HTTP response headers: HTTPHeaderDict({'Date': 'Sun, 30 Aug 2020 16:51:15 GMT', 'Content-Type': 'application/json', 'Content-Length': '172', 'Connection': 'keep-alive', 'Vary': 'Origin,Accept-Encoding'}) HTTP response body: {"error":"Company not found. The data may not be available in the Sandbox environment.","message":"An error occured. Please contact success@intrinio.com with the details."}
# 
# has limitation # of calls per minute
# 
# Reason: Too Many Requests
# HTTP response headers: HTTPHeaderDict({'Date': 'Tue, 08 Sep 2020 06:24:51 GMT', 'Content-Type': 'application/json', 'Content-Length': '233', 'Connection': 'keep-alive', 'Vary': 'Origin,Accept-Encoding'})
# HTTP response body: {"error":"1 Minute Call Limit Reached","message":"You have exceeded the limits for how frequently you can make API calls with high paging parameters. Please contact support if you need a higher limit.","access_codes":["high_paging"]}

# # https://plotly.com/python/filled-area-plots/
# 
# between area: same color, but no different colors

# # execution method
# - Download visualization_code folder
# - Execute “visualization.ipynb”

# In[1]:


#!pip install dash

import dash ## local 
#from jupyter_dash import JupyterDash # colab
import dash_html_components as html
import dash_core_components as dcc
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from dash.dependencies import Input, Output

from sklearn import preprocessing
import numpy as np
import tensorflow as tf

import yfinance as yf


# In[2]:


stock_list = ["FB", "AAPL", "AMZN", "NFLX", "GOOG", "MSFT", "XLK", "QQQ"]


# In[3]:


stock_dic = dict()

for stock in stock_list:
    df = yf.download(stock, start="2019-7-25").drop(columns='Adj Close')
    df.dropna(inplace=True)
    stock_dic[stock] = df


# # predict1, predict2, predict3, predict30, MA, PMA, UpDown

# In[4]:


history_points = 50
def getXtest(data):
    data = data.to_numpy()
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data)
    Xtest = np.array([data_normalised[i : i + history_points].copy() for i in range(len(data) - history_points+1)])

    next_day_open_values = np.array([data[:,0][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_open_values = np.expand_dims(next_day_open_values, -1)
  
    next_day_close_values = np.array([data[:,3][i + history_points].copy() for i in range(len(data) - history_points)])
    next_day_close_values = np.expand_dims(next_day_close_values, -1)
    
    y_normaliser_open = preprocessing.MinMaxScaler()
    y_normaliser_open.fit(next_day_open_values)
    
    y_normaliser_close = preprocessing.MinMaxScaler()
    y_normaliser_close.fit(next_day_close_values)

    return Xtest, y_normaliser_open, y_normaliser_close


# In[5]:


# 1 day prediction: close, open (for candle stick)
def getResult1(stock):
    data = stock_dic[stock]
    
    model_open = tf.keras.models.load_model('{}_model_open'.format(stock))
    model_close = tf.keras.models.load_model('{}_model'.format(stock))
    
    Xtest, y_normaliser_open, y_normaliser_close = getXtest(data)

    open_pred = model_open.predict(Xtest)
    open_pred = y_normaliser_open.inverse_transform(open_pred)
    
    close_pred = model_close.predict(Xtest)
    close_pred = y_normaliser_close.inverse_transform(close_pred)
    
    # return open_pred, close_pred
    return np.reshape(open_pred, open_pred.shape[0]), np.reshape(close_pred, close_pred.shape[0])


# In[6]:


def getResult_days(stock, prediction_days):

    data = stock_dic[stock]
    
    data1 = data.to_numpy()
    data_normaliser = preprocessing.MinMaxScaler()
    data_normalised = data_normaliser.fit_transform(data1)

    model = tf.keras.models.load_model('{}_model_'.format(stock)+str(prediction_days)+'days')
    
    Xtest, y_normaliser_open, y_normaliser = getXtest(data)
    
    y_test_predicted = model.predict(Xtest)
    y_test_predicted = y_normaliser.inverse_transform(y_test_predicted)
    
    y_test_prediction = []
    for i in y_test_predicted[:-1]:
        y_test_prediction.append(i[0])
    
    y_pred = y_test_prediction + list(y_test_predicted[-1])
    
    if prediction_days != 2:
        return y_pred
    
    else: # add PMA, MA if prediction_days = 2
        one_day_data = []
        day4_data = []
        ma = []

        period = len(data)-history_points
        for i in range(period-1,-1,-1):
            ma.append(sum(data['Close'].iloc[-i-7:-i-1])/6)
            one_day_data.append(data_normalised[-i-51:-i-1])
            day4_data.append(sum(data['Close'].iloc[-i-5:-i-1]))
        one_day_data = np.array(one_day_data)

        one_day_predict = model.predict(one_day_data)
        one_day_predicted = y_normaliser.inverse_transform(one_day_predict)

        day6_sum = []
        for i in range(period):
            s = day4_data[i] + one_day_predicted[i][0] + one_day_predicted[i][1]
            day6_sum.append(s/6)

        PMA = [day6_sum[i] for i in range(period)]
        MA = [ma[i] for i in range(period)]

        UpDown = []
        for i in range(len(PMA)):
            if PMA[i] >= MA[i]:
                UpDown.append(1)
            else:
                UpDown.append(0)

        return y_pred, PMA, MA, UpDown


# In[7]:


df_dic = dict()

for stock in stock_list:
    result_frame = stock_dic[stock].copy().loc[stock_dic[stock].index[history_points]:]
    idx = pd.date_range(result_frame.index[-1], periods=31, freq='B')[1:]
    open_pred, close_pred = getResult1(stock)

    pred2, PMA, MA, UpDown = getResult_days(stock, 2)
    pred3 = getResult_days(stock, 3)
    pred30 = getResult_days(stock, 30)


    for i, date in enumerate(idx):
    # Open High Low Close Volume
#         if i ==0:
#             openpp = open_pred[-1]
#             closepp = close_pred[-1]
#             result_frame.loc[pd.to_datetime(date)] = [open_pred[-1], max(openpp,closepp), min(openpp,closepp), close_pred[-1], None]
#         else:
        result_frame.loc[pd.to_datetime(date)] = [None, None, None, None, None]
    
    
    result_frame['Open_Pred'] = list(open_pred) + [None for i in range(len(result_frame)-len(open_pred))]
    result_frame['Close_Pred1'] = list(close_pred) + [None for i in range(len(result_frame)-len(close_pred))]
    result_frame['Close_Pred2'] = list(pred2) + [None for i in range(len(result_frame)-len(pred2))]
    result_frame['Close_Pred3'] = list(pred3) + [None for i in range(len(result_frame)-len(pred3))]
    result_frame['Close_Pred30'] = pred30
    result_frame['PMA'] = PMA + [None for i in range(len(result_frame)-len(PMA))]
    result_frame['MA'] = MA + [None for i in range(len(result_frame)-len(MA))]
    result_frame['UpDown'] = UpDown + [None for i in range(len(result_frame)-len(UpDown))]

    
    close = stock_dic[stock].Close[history_points-1:]

    colors = list()
    for i in range(1, len(close)):
        if close[i-1] <= close[i]:
            colors.append('green')
        else:
            colors.append('red')        
    
    result_frame['colors'] = colors + ['red' for i in range(len(result_frame)-len(colors))]    
    
    df_dic[stock] = result_frame


# In[8]:


today = stock_dic['FB'].index[-1]
next_date30 = df_dic['FB'].index[-1]
next_days = pd.date_range(today, periods=4, freq='B')
next_date = next_days[1]
next_date3 = next_days[-1]
# daysBTW = int(str(next_date3-today)[:2])+1


# In[9]:


print(today)
print(next_date30)
# print(daysBTW)
print(next_date)
print(next_date3)


# In[10]:


candle_dic = dict()
for stock in stock_list:
    Open = list()
    Close = list()
    High = list()
    Low = list()
    result_frame = df_dic[stock].copy()
    for date in result_frame.index:
        if date != next_date:
            Open.append(result_frame.loc[date].Open)
            Close.append(result_frame.loc[date].Close)
            High.append(result_frame.loc[date].High)
            Low.append(result_frame.loc[date].Low)
        else:
            Open.append(result_frame.loc[next_date].Open_Pred)
            Close.append(result_frame.loc[next_date].Close_Pred1)
            High.append(max(result_frame.loc[next_date].Open_Pred,result_frame.loc[next_date].Close_Pred1))
            Low.append(min(result_frame.loc[next_date].Open_Pred,result_frame.loc[next_date].Close_Pred1))
    df = pd.DataFrame(data={'Open': Open, "High":High,"Low":Low,'Close':Close})
    df = df.set_index(result_frame.index)
    candle_dic[stock] = df


# In[11]:


options_list = [
    {'label': 'Apple', 'value': 'AAPL'},
    {'label': 'Amazon', 'value': 'AMZN'},
    {'label': 'Facebook', 'value': 'FB'},
    {'label': 'Google', 'value': 'GOOG'},
    {'label': 'Microsoft', 'value': 'MSFT'},
    {'label': 'Neflix', 'value': 'NFLX'},
    {'label': 'XLK', 'value': 'XLK'}, 
    {'label': 'QQQ', 'value': 'QQQ'}    
]


# In[12]:


date_options_list = [
    {'label': '2 Days', 'value': -5},
    {'label': '5 Days', 'value': -8},
    {'label': '1 Month', 'value': -27},
    {'label': '3 Month', 'value': -69},
    {'label': '6 Month', 'value': -133},
    {'label': 'All', 'value': 'all'}  
]


# In[13]:


# Initialize the app
app = dash.Dash(__name__) ## local
#app = JupyterDash(__name__) # colab
app.config.suppress_callback_exceptions = True


# In[14]:


app.layout = html.Div([
# Setting the main title of the Dashboard
    html.H1("Stock Price Prediction", style={"textAlign": "center"}),
    
    html.Div([
        html.H1("Please select stock", 
            style={'textAlign': 'center'}),
            # Adding the first dropdown menu and the subsequent time-series graph
            dcc.Dropdown(id='stock_select',
                options=options_list, # multi=True,
                value='FB',
                style={"display": "block", "margin-left": "auto", 
                    "margin-right": "auto", "width": "60%"}),
        
            dcc.Dropdown(id='date_select',
                options=date_options_list, # multi=True,
                value=-8,
                style={"display": "block", "margin-left": "auto", 
                    "margin-right": "auto", "width": "60%"}),
                dcc.Graph(id='basicGraph'),
                dcc.Graph(id='volumeGraph'),
                dcc.Graph(id='predictionGraph'),
        
            ])
])


# In[15]:


@app.callback(Output('basicGraph', 'figure'),
              [Input('stock_select', 'value'), Input('date_select', 'value')])

def update_graph(selected_dropdown, selected_date):

    trace1 = []
    trace2 = []
    trace3 = []
    trace4 = []
    trace5 = []
    trace6 = []
    trace7 = []
    trace8 = []
    trace9 = []
    
    if selected_date == 'all':
        result_frame = df_dic[selected_dropdown].copy()
        candle_frame = candle_dic[selected_dropdown].copy()
    else:
        result_frame = df_dic[selected_dropdown].copy().loc[:next_date3].iloc[selected_date:]
        candle_frame = candle_dic[selected_dropdown].copy().loc[:next_date3].iloc[selected_date:]

    trace1.append(
      go.Candlestick(x=candle_frame.index, visible='legendonly',
                     open=candle_frame.Open,
                     high=candle_frame.High,
                     low=candle_frame.Low,
                     close=candle_frame.Close,
                     name=f'Candlestick'))

    trace2.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.MA,
                 mode='lines', opacity=0.8, #fill="tonexty",
                 name=f'MA', #,textposition='bottom center',
                line = dict(color='chocolate')))
    
    trace3.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.PMA,
                 mode='lines', opacity=0.9, #fill='tonexty',
                 name=f'PMA',#textposition='bottom center',
                 line = dict(color='blue')))
    
    trace4.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Open, visible='legendonly',
                 mode='lines', opacity=0.8, line=dict(color="orange", width=4, dash='dot'),
                 name=f'Open')) #,textposition='bottom center'))
    
    trace5.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close, visible='legendonly',
                 mode='lines', opacity=0.6, line=dict(color="teal", width=4, dash='dash'),
                 name=f'Close')) #,textposition='bottom center'))
    
    trace6.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred1, visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "cyan",
                 name=f'Predict 1day')) #,textposition='bottom center'))
    trace7.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred2, visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "magenta",
                 name=f'Predict 2days')) #,textposition='bottom center'))
    
    trace8.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred3, visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "navy",
                 name=f'Predict 3days')) #,textposition='bottom center'))
    
    trace9.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred30, visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "black",
                 name=f'Predict 30days')) #,textposition='bottom center'))
     
    traces = [trace1, trace2, trace3, trace4, trace5, trace6, trace7, trace8, trace9]
 
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout( #colorway=["#5E0DAC", '#800000', '#FFA500', 
                                    #        '#00FFFF', '#FF00FF','#0000FF'],
            height=700, 
            legend=dict(
                orientation="h",              
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
             title=f"{selected_dropdown}", xaxis={'rangeslider': {'visible': False}, 'type': 'date'},
#             yaxis =  {"title":"Price", "range":[min(result_frame.Open)*0.75,max(result_frame.Open*1.1)],
#                       'fixedrange': False},               
      
                  
#             yaxis2={"title":"Volume", "side":"right", "overlaying":"y", 
#                     "range":[min(result_frame.Volume),max(result_frame.Volume)*4]},                                                  
                  
                  
             shapes=[
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=next_date+pd.DateOffset(-1),
                        y0="0",
                        x1=result_frame.index[-1],
                        y1="1",
                        fillcolor="lightgray",
                        opacity=0.4,
                        line_width=0,
                        layer="below"
                    ),
                    ],  
                  
# #             xaxis={#"title":"Date",
# # #                    'rangeselector': {'buttons': list([
# # #                        {'count': daysBTW, 'label': '1D', 'step': 'day','stepmode': 'backward'},
# # #                        {'count': daysBTW+6, 'label': '5D', 'step': 'day', 'stepmode': 'backward'},                       
# # #                        {'count': daysBTW+30, 'label': '1M', 'step': 'day', 'stepmode': 'backward'},
# # #                        {'count': daysBTW+92, 'label': '3M', 'step': 'day', 'stepmode': 'backward'},
# # #                        {'count': daysBTW+183, 'label': '6M', 'step': 'day','stepmode': 'backward'},
# # #                        {'step': 'all'}])},                
# #                    'rangeslider': {'visible': False}, 
# #                    'type': 'date'}, 
                  

              )}    
    
    return figure


# In[16]:


@app.callback(Output('volumeGraph', 'figure'),
              [Input('stock_select', 'value'), Input('date_select', 'value')])

def update_graph2(selected_dropdown, selected_date):

    trace10 = []

    
    if selected_date == 'all':
        result_frame = df_dic[selected_dropdown].copy()
    else:
        result_frame = df_dic[selected_dropdown].copy().loc[:next_date3].iloc[selected_date:]


   
    trace10.append(
      go.Bar(x=result_frame.index, y=result_frame.Volume,opacity=0.7, 
                      name=f'Volume',marker_color=result_frame.colors)) 

  
    traces = [trace10]
 
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout( #colorway=["#5E0DAC", '#800000', '#FFA500', 
                                    #        '#00FFFF', '#FF00FF','#0000FF'],
            height=400, 
            legend=dict(
                orientation="h",              
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
             title=f"{selected_dropdown} Volume", 
             xaxis={'rangeslider': {'visible': False}, 'type': 'date'},
                  
#             yaxis =  {"title":"Price", "range":[min(result_frame[:-30].Open)*0.75,max(result_frame[:-30].Open*1.1)],
#                       'fixedrange': False},
                
                  shapes=[
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=next_date+pd.DateOffset(-1),
                        y0="0",
                        x1=result_frame.index[-1],
                        y1="1",
                        fillcolor="lightgray",
                        opacity=0.4,
                        line_width=0,
                        layer="below"
                    ),
                    ],            
                  
#             yaxis2={"title":"Volume", "side":"right", "overlaying":"y", 
#                     "range":[min(result_frame[:-30].Volume),max(result_frame[:-30].Volume)*4]},
              )}    
                  
   
    return figure
        
        
        


# In[17]:


@app.callback(Output('predictionGraph', 'figure'),
              [Input('stock_select', 'value'), Input('date_select', 'value')])
def update_graph3(selected_dropdown, selected_date):

    trace6 = []
    trace7 = []
    trace8 = []
    trace9 = [] 
    
    trace11 = []    
    
    if selected_date == 'all':
        result_frame = df_dic[selected_dropdown].copy()
    else:
        result_frame = df_dic[selected_dropdown].copy().loc[:next_date3].iloc[selected_date:]


    trace6.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred1, #visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "cyan",
                 name=f'Predict 1day')) #,textposition='bottom center'))
    trace7.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred2, #visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "magenta",
                 name=f'Predict 2days')) #,textposition='bottom center'))
    
    trace8.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred3, #visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "navy",
                 name=f'Predict 3days')) #,textposition='bottom center'))
    
    trace9.append(
      go.Scatter(x=result_frame.index,
                 y=result_frame.Close_Pred30, visible='legendonly',
                 mode='lines', opacity=0.8, line_color= "black",
                 name=f'Predict 30days')) #,textposition='bottom center'))

    
    # next day candle: 'Open_Pred', 'Close_Pred1',
    Open = list()
    Close = list()
    High = list()
    Low = list()
    for date in result_frame.index:
        if date != next_date:
            Open.append(None)
            Close.append(None)
            High.append(None)
            Low.append(None)
        else:
            Open.append(result_frame.loc[next_date].Open_Pred)
            Close.append(result_frame.loc[next_date].Close_Pred1)
            High.append(max(result_frame.loc[next_date].Open_Pred,result_frame.loc[next_date].Close_Pred1))
            Low.append(min(result_frame.loc[next_date].Open_Pred,result_frame.loc[next_date].Close_Pred1))
    
    trace11.append(
      go.Candlestick(x=result_frame.index, visible='legendonly',
                     open=Open,
                     high=High,
                     low=Low,
                     close=Close, opacity=0.7, 
                     #increasing_line_color= 'green', decreasing_line_color= 'red',
                     name=f'next candle'))  
  
    traces = [trace6, trace7, trace8, trace9, trace11]
 
    data = [val for sublist in traces for val in sublist]
    figure = {'data': data,
              'layout': go.Layout( #colorway=["#5E0DAC", '#800000', '#FFA500', 
                                    #        '#00FFFF', '#FF00FF','#0000FF'],
            height=400, 
            legend=dict(
                orientation="h",              
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1),
             #title=f"{selected_dropdown}", 
             xaxis={'rangeslider': {'visible': False}, 'type': 'date'},
                  
#             yaxis =  {"title":"Price", "range":[min(result_frame[:-30].Open)*0.75,max(result_frame[:-30].Open*1.1)],
#                       'fixedrange': False},
                
                  shapes=[
                    dict(
                        type="rect",
                        xref="x",
                        yref="paper",
                        x0=next_date+pd.DateOffset(-1),
                        y0="0",
                        x1=result_frame.index[-1],
                        y1="1",
                        fillcolor="lightgray",
                        opacity=0.4,
                        line_width=0,
                        layer="below"
                    ),
                    ],            
                  
#             yaxis2={"title":"Volume", "side":"right", "overlaying":"y", 
#                     "range":[min(result_frame[:-30].Volume),max(result_frame[:-30].Volume)*4]},
              )}    
                  
   
    return figure


# In[18]:


# Run the app

#app.run_server(mode='inline')  # colab
if __name__ == '__main__':
    app.run_server()


# WARNING: Do not use the development server in a production environment.
# 
# Is only a warning. Then means, if you run your site in the world wide web... then don't activate the developement mode on THAT specific server, for security reasons.

# In[ ]:




