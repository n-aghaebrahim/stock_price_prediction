import os

import matplotlib.pyplot as plt
from pandas.plotting import lag_plot, autocorrelation_plot

def lag_fig(stock_data, stock_name, date_info, time):
    if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/data_plots'):
        os.makedirs(f'logs/{stock_name}/{date_info}/{time}/data_plots')

    lag_plot(stock_data)
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/data_plots/laag_plot.png')
    plt.clf()


def autocorrelation(stock_data, stock_name, date_info, time):
    if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/data_plots'):
        os.makedirs(f'logs/{stock_name}/{date_info}/{time}/data_plots')

    autocorrelation_plot(stock_data)
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/data_plots/aautocorrelation_plot.png')
    plt.clf()


def plot_fig(history,s, stock_name, date_info, time, update_condition, predict_date, predict_time, n_update_epoch):
    
    plot_acc_name = f'acc_sequance_{s}'
    plot_loss_name = f'loss_sequance_{s}'
    if update_condition:
        time = predict_time
        date_info = predict_date
        plot_acc_name = plot_acc_name + f'_update_{n_update_epoch}'
        plot_loss_name = plot_loss_name + f'_update_{n_update_epoch}'

    if not os.path.exists(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots'):
        os.makedirs(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots')
    plt.clf()


    plt.plot(history.history['accuracy'])
    plt.plot(history.history['val_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots/{plot_acc_name}.jpg')

    plt.clf()

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'val'], loc='upper left')
    plt.savefig(f'logs/{stock_name}/{date_info}/{time}/seq_{s}/plots/{plot_loss_name}.jpg')
