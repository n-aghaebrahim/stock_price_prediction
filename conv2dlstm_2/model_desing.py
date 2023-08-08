import os
import json
import math
import plot
import pathlib

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

from numpy import array
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from datetime import date, timedelta, datetime
from keras.preprocessing.sequence import TimeseriesGenerator
from contextlib import redirect_stdout

from indicators import INDICATOR


pd.options.mode.chained_assignment = None


# Function to create a model
def MODEL(n_steps, 
          n_features, 
          about_model_summary, 
          filter_num, 
          dilation, 
          kernel, 
          pad, 
          pool, 
          lstm_layer_dict, 
          stock_name,
          date_info,
          time,
          lstm_n_last_layer,
          last_layer_dropout,
          train_condition
          ):

    if train_condition:
        if not pathlib.Path(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}').is_dir():
            pathlib.Path('logs/{stock_name}/{date_info}/{time}/seq_{n_steps}').mkdir(parents=True)

        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}/info_dict.json', "w") as outfile:
            json.dump(about_model_summary, outfile, indent=6)

        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}/info.txt', 'a') as filet:
            filet.write('\n--------------------\n')
            filet.write('model information\n')
            filet.write('---------------------\n')
            filet.write(f'filter_num: {filter_num}\ndilation_rate: {dilation}\nkernel_size: {kernel}\npading: {pad}\npool_size: {pool}\nlstm_num: {str(lstm_layer_dict)}\n')
            #filet.write(about_model_summary)
        
    # Define a placeholder for the input data
    from keras.layers import Reshape
    inputs = tf.keras.layers.Input(shape=(n_steps, n_features))
    inputs = Reshape((n_steps,1, n_features, 1))(inputs)
    #print(inputs)
    
    # Apply a TimeDistributed layer to the input
    #x = tf.keras.layers.TimeDistributed(tf.keras.layers.Conv2D(32, (3, 1), activation='relu'))(inputs)
    

    #tensor_input = layers.Input(shape=(None, n_steps, n_features,1), name='main_inputs')
    cnn = layers.TimeDistributed(tf.keras.layers.Conv2D(filters=filter_num, kernel_size=kernel, dilation_rate=dilation, padding=pad, activation='relu'))(inputs)#(tensor_input)
    #cnn = layers.TimeDistributed(layers.Conv2D(filters=filter_num, kernel_size=kernel, dilation_rate=dilation, padding=pad, activation='relu'))(cnn)
    #cnn = layers.TimeDistributed(layers.MaxPooling2D(pool_size=pool))(cnn)
    
    cnn = layers.TimeDistributed(layers.Flatten())(cnn)
    
    lstm_layer_list = []
    i = 0

    for k, v in lstm_layer_dict.items():
        if i == 0:
            lstm = layers.LSTM(v, activation='relu', return_sequences= True)(cnn)
            lstm = layers.Dropout(0.2)(lstm)

        else:
            lstm = layers.LSTM(v, activation='relu', return_sequences= True)(lstm)
            lstm = layers.Dropout(0.2)(lstm)

        i +=1 
        
    lstm_out = layers.LSTM(lstm_n_last_layer, activation='relu')(lstm)
    lstm_out = layers.Dropout(last_layer_dropout)(lstm_out)

    out = layers.Dense(4)(lstm_out)

    model = keras.Model(inputs=inputs, outputs=out)

    
    #model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mae', metrics = ['accuracy'])
    print('\n---------------------------')
    print('     * model summary *     ')
    print('---------------------------')

    if train_condition:
        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{n_steps}/info_s.txt', 'w') as filet:
            with redirect_stdout(filet):
                model.summary()
   
    print('model is created')
    print(model.summary())
    return model
   

