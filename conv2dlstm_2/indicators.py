import os
import json
import math

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import yfinance as yf

from numpy import array
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from keras.preprocessing.sequence import TimeseriesGenerator
from datetime import date, timedelta, datetime
from contextlib import redirect_stdout

pd.options.mode.chained_assignment = None



class INDICATOR:

    
    # calculate on_balance_volume
    def obv(self, stock_data, span_period:list):

        obv = []
        obv.append(0)

        for i in range(1, len(stock_data['Close'])):
            if stock_data['Close'][i] > stock_data['Close'][i-1]:
                obv.append(obv[-1] + stock_data['Volume'][i])

            elif stock_data['Close'][i] < stock_data['Close'][i-1]:
                obv.append(obv[-1] - stock_data['Volume'][i])

            else:
                obv.append(obv[-1])

        stock_data['obv'] = obv
        
        for i in span_period:
            stock_data[f'obv_ema{i}'] = stock_data['obv'].ewm(span=i).mean()

        return stock_data


    # average directional index helper funciton
    def adx(self, high, low, close, lookback):
        plus_dm = high.diff()
        minus_dm = low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm > 0] = 0

        tr1 = pd.DataFrame(high - low)
        tr2 = pd.DataFrame(abs(high - close.shift(1)))
        tr3 = pd.DataFrame(abs(low - close.shift(1)))
        frames = [tr1, tr2, tr3]
        tr = pd.concat(frames, axis = 1, join = 'inner').max(axis = 1)
        atr = tr.rolling(lookback).mean()

        plus_di = 100 * (plus_dm.ewm(alpha = 1/lookback).mean() / atr)
        minus_di = abs(100 * (minus_dm.ewm(alpha = 1/lookback).mean() / atr))
        dx = (abs(plus_di - minus_di) / abs(plus_di + minus_di)) * 100
        adx = ((dx.shift(1) * (lookback - 1)) + dx) / lookback
        adx_smooth = adx.ewm(alpha=1/lookback).mean()

        return plus_di, minus_di, adx_smooth

    
    # Calculate the average diretional index
    def get_adx(self, stock_data):
        stock_data['plus_di'] = pd.DataFrame(self.adx(stock_data['High'], 
                                             stock_data['Low'], 
                                             stock_data['Close'], 14)[0]).rename(columns = {0: 'plus_di'})

        stock_data['minus_di'] = pd.DataFrame(self.adx(stock_data['High'], 
                                              stock_data['Low'], 
                                              stock_data['Close'], 14)[1]).rename(columns = {0: 'minus_di'})

        stock_data['adx'] = pd.DataFrame(self.adx(stock_data['High'], 
                                                  stock_data['Low'], 
                                                  stock_data['Close'], 14)[2]).rename(columns = {0: 'adx'})

        stock_data = stock_data.dropna()

        return stock_data


    # Calculate the accumulation/distribution indicator
    def a_d(self, stock_data):
        stock_data['mfm'] = ((stock_data['Close'] - stock_data['Low']) - (stock_data['High'] - stock_data['Close'])) / (stock_data['High'] - stock_data['Low'])

        stock_data['mfv'] = stock_data['mfm'] * stock_data['Volume']

        ad = []
        ad.append(0)

        for i in range(1, len(stock_data['Close'])):
            ad.append(ad[-1] + stock_data['mfv'][i])

        stock_data['a/d'] = ad

        return stock_data


    # Calculate the sxponential moving average
    def ewm(self, stock_data, span_list: list):
        
        for i in span_list:
            stock_data[f'ewm{i}'] = stock_data['Close'].ewm(span=i, adjust = False).mean()


        return stock_data


    # Calculate the moving average convergance divergence (MACD)
    def macd(self, stock_data, span1: int=12, span2: int=26):
        stock_data[f'ewm{span1}'] = stock_data['Close'].ewm(span=span1, adjust=False).mean()
        stock_data[f'ewm{span2}'] = stock_data['Close'].ewm(span=span2, adjust=False).mean()
        stock_data['macd'] = stock_data[f'ewm{span1}'] - stock_data[f'ewm{span2}']

        stock_data = stock_data.drop(f'ewm{span1}', axis=1)
        stock_data = stock_data.drop(f'ewm{span2}', axis=1)

        return stock_data

    
    # Calculate the stochastic osillator indicator
    def soi(self, stock_data, span: int=14):
        stock_data[f'high_{span}'] = stock_data['High'].rolling(span).max()
        stock_data[f'low_{span}'] = stock_data['Low'].rolling(span).min()

        stock_data['k'] = (stock_data['Close'] - stock_data[f'low_{span}']) * 100 / (stock_data[f'high_{span}'] - stock_data[f'low_{span}'])

        stock_data['d'] = stock_data['k'].rolling(3).mean()

        stock_data = stock_data.drop(f'high_{span}', axis=1)
        stock_data = stock_data.drop(f'low_{span}', axis=1)

        stock_data['k'] = stock_data['k'].fillna(0)
        stock_data['d'] = stock_data['d'].fillna(0)
        stock_data = stock_data.fillna(0)

        return stock_data

    
    # calculate the relative strength index (rsi)
    def rsi(self, df, periods = 14, ema = True):
        """
        Returns a pd.Series with the relative strength index.
        """
        close_delta = df['Close'].diff()

        # Make two series: one for lower closes and one for higher closes
        up = close_delta.clip(lower=0)
        down = -1 * close_delta.clip(upper=0)
    
        if ema == True:
	    # Use exponential moving average
            ma_up = up.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
            ma_down = down.ewm(com = periods - 1, adjust=True, min_periods = periods).mean()
        else:
            # Use simple moving average
            ma_up = up.rolling(window = periods, adjust=False).mean()
            ma_down = down.rolling(window = periods, adjust=False).mean()
        
        rsi = ma_up / ma_down
        rsi = 100 - (100/(1 + rsi))
        
        df['rsi'] = rsi
        df = df.dropna()   

        return df


    # Calculate the sma
    def get_sma(self, prices, rate):
        return prices.rolling(rate).mean()


    def get_bollinger_bands(self, stock_price, rate=20):
        stock_price_close = stock_price['Close']
        sma = self.get_sma(stock_price_close, rate)
        std = stock_price_close.rolling(rate).std() # <-- Get rolling standard deviation for 20 days

        bollinger_up = sma + std * 2 # Calculate top band
        bollinger_down = sma - std * 2 # Calculate bottom band

        stock_price['bollinger_up'] = bollinger_up
        stock_price['bollinger_down'] = bollinger_down

        stock_price = stock_price.dropna()

        return stock_price

