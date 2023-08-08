import os
import json
import math
import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from numpy import array
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.preprocessing import MinMaxScaler
from multiprocessing import Pool
from datetime import date, timedelta, datetime
from keras.preprocessing.sequence import TimeseriesGenerator
from contextlib import redirect_stdout
pd.options.mode.chained_assignment = None

import plot
from indicators import INDICATOR





def model_train(symbol, 
                model, 
                x_train, 
                y_train, 
                x_test, 
                y_test,  
                epochs, 
                train_condition, 
                predict_condition, 
                update_condition, 
                s,
                stock_name,
                date_info,
                time,
                predict_time,
                predict_date,
                learning_rate_update,
                learning_rate,
                validation_split,
                batch_size,
                predict_weight,
                best_weight,
                best_weight_update,
                use_update_weight_update,
                n_update_epoch,
                validation_split_update):

        
    if update_condition:
        print("\n####################################################")
        print(f"start to updata model for sequence of {s}.....")
        print("####################################################")
        if best_weight_update and not use_update_weight_update:
            bp = 'best_'

        else:
            bp = ''

        if use_update_weight_update and not best_weight_update:
            up = '_updated_{n_update_epoch}'

        else:
            up = ''
        model.load_weights(f'logs/{stock_name}/{predict_date}/{predict_time}/seq_{s}/weights/{bp}weight{up}_epoch_{epochs}.h5')
        print('pre weights are loaded...')
        model.compile(optimizer= tf.keras.optimizers.lagacy.Adam(learning_rate = learning_rate_update) , loss='mse', metrics = ['accuracy'])
        print('update the model with the new data')
        history = model.fit(x_train, y_train, epochs=n_update_epoch, validation_split = validation_split_update, shuffle=False, batch_size= batch_size)

        # plot the training history
        plot.plot_fig(history, s, stock_name, date_info, time, update_condition, predict_date, predict_time, n_update_epoch)
        model.save(f'logs/{stock_name}/{predict_date}/{predict_time}/seq_{s}/weights/weight_updated_{n_update_epoch}_epoch_{epochs}.h5')

    #elif os.path.isfile(f'{symbol}_weights/weight_10.h5'):
    if predict_condition:
        print("\n####################################################")
        print(f"start to make prediction for sequence of {s}.....")
        print("####################################################")
        if best_weight:
            bp = 'best_' 
        else:
            bp = ''
        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate) , loss='mse', metrics = ['accuracy'])
        print(f'logs/{stock_name}/{predict_date}/{predict_time}/seq_{s}/weights/{bp}weight_{predict_weight}epoch_{epochs}.h5') 
        model.load_weights(f'logs/{stock_name}/{predict_date}/{predict_time}/seq_{s}/weights/{bp}weight_{predict_weight}epoch_{epochs}.h5') 
        print('pre weights are loaded...')

    if train_condition:
        print("\n####################################################")
        print(f"start to train model for sequence of {s}.....")
        print("####################################################")
        model.compile(optimizer= tf.keras.optimizers.Adam(learning_rate = learning_rate) , loss='mse', metrics = ['accuracy'])
 
        if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/weights'):
            os.makedirs(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/weights')         

        checkpoint_path = f'logs/{stock_name}/{date_info}/{time}/seq_{s}/weights/best_weight_epoch_{epochs}.h5'
        model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
                                                                    filepath=checkpoint_path,
                                                                    save_weights_only=True,
                                                                    monitor='val_loss',
                                                                    mode='min',
                                                                    save_best_only=True)

        #print(x_train.shape)
        #exit()
        #x_train = x_train.reshape((x_train.shape[0],1,x_train.shape[1],x_train.shape[2], x_train.shape[3]))
        #print(x_train)
        history = model.fit(x_train, y_train, shuffle=False, validation_split= validation_split, epochs=epochs, batch_size=batch_size, callbacks=[model_checkpoint_callback])

        # plot the training history
        plot.plot_fig(history, s, stock_name, date_info, time, update_condition, predict_date, predict_time, n_update_epoch)
        
        # reading the training history and save it to .json file
        new_dict = {}
        for k, v in history.history.items():
            for i in range(len(v)):    
                new_dict[i] = {}

        for k, v in history.history.items():
            for i in range(len(v)):
            #for k, v in history.history.items():
                    new_dict[i][k] = v[i]

        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/train_result.json', 'w') as tr:
            json.dump(new_dict, tr, indent=4)
            
        print('\n------------------------------------')
        print('evaluate the model on the test data:')
        print('------------------------------------')
        e = model.evaluate(x_test, y_test, batch_size=1)
        print('model evaluate result is\n')
        print(e)

        with open(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/info.txt', 'a') as filet:
            filet.write('\n-------------------\n')
            filet.write('the test result')
            filet.write('------------------\n')
            filet.write(f'the loss on test set is: {e[0]}\n')
            filet.write(f'the accuracy on test set is: {e[1]}\n')

        if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/weights'):
            os.makedirs(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/weights')  

        model.save(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/weights/weight_epoch_{epochs}.h5')

        if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/models'):
            os.makedirs(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/models')

        model.save(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/models/model_{epochs}')
    return model



#def prediction(model, x_test, y_test, scaler, stock_data, training_data_len):
def prediction(model, scaled_predict, y_predict, stock_data, ypscaler):
    y_pred_scaled = model.predict(scaled_predict)
    y_pred = ypscaler.inverse_transform(y_pred_scaled)
    
    print(len(y_pred))

    df = pd.DataFrame(y_pred, columns = ['Open', 'High', 'Low', 'Close'])#, 'Volume', 'ewm5', 'ewm10'])#, 'Adj Close', 'Volume', 'ewm5', 'ewm10', 'ewm20', 'ewm50', 'ewm200', 'macd', 'k', 'd'])  
    print(df.tail())
    
    return df 


