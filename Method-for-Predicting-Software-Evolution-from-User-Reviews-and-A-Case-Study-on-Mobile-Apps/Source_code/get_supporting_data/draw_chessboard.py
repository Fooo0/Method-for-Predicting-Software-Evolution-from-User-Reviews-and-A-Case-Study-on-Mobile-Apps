# encoding: utf-8
'''
Created on 2019.03.17

@author: Molly Song
'''


import os
import numpy
import pandas
from pylab import mpl
import matplotlib.pyplot as plt


SERIAL_DIR = 'D:\programming\workspacePycharm\masterProject\Data\Serial_predict'


MIN_PERIOD = 2


def filter_update_days(days):
    num_days = len(days)
    days.sort()
    return [days[i] for i in range(num_days) if i - 1 < 0 or days[i] - days[i - 1] > MIN_PERIOD]


def get_update_day(file_name):
    return int(file_name.split('.')[0])


def get_chess_data_content(data_path):
    df = pandas.read_excel(data_path)
    features = df['feature'].tolist()

    df = df.drop(columns=['feature'])
    days = df.columns.values.tolist()
    arr = numpy.array(df.values)
    return features, days, arr


def get_chess_data_time(data_path):
    df = pandas.read_excel(data_path)
    days = df['day'].tolist()
    arr = numpy.array([df['prediction'].tolist(),])
    return days, arr


def draw_chessboard(chess_data, cm, x_tick_labels, u_days, y_tick_labels, y_label, app_name, svg_path):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    ax.imshow(chess_data, cm, aspect='equal', interpolation='nearest', origin='lower')

    ax.set_xticks(numpy.arange(chess_data.shape[1]))
    ax.set_yticks(numpy.arange(chess_data.shape[0]))
    ax.set_xlim(-0.5, )

    print(u_days)
    _x_labels = []
    for item in x_tick_labels:
        if item not in u_days and (item-MIN_PERIOD) in u_days:
            _x_labels.append(item)
        else:
            _x_labels.append('')

    ax.set_xticklabels(_x_labels, fontsize=3)
    ax.set_yticklabels(y_tick_labels, fontsize=2)
    ax.set_xlabel('Timeline:the t-th Day from 2016.3.1', fontsize=4)
    ax.set_ylabel(y_label, fontsize=4)

    # plt.setp(ax.get_xticklabels(), rotation=-30, ha="right", rotation_mode="anchor")
    plt.title(app_name, fontsize=7)
    ax.set_xticks(numpy.arange(chess_data.shape[1] + 1) - 0.5, minor=True)
    ax.set_yticks(numpy.arange(chess_data.shape[0] + 1) - 0.5, minor=True)
    ax.grid(which='minor', linewidth=0.2)
    ax.tick_params(which='minor', bottom=False, left=False)
    fig.tight_layout()

    ax.spines['bottom'].set_linewidth(0.5)
    ax.spines['left'].set_linewidth(0.5)
    ax.spines['right'].set_linewidth(0.5)
    ax.spines['top'].set_linewidth(0.5)
    ax.tick_params(axis='x', width=0.5)
    ax.tick_params(axis='y', width=0.5)

    plt.savefig(svg_path, format='svg')
    plt.close()


if __name__ == '__main__':
    '''
    category = 'Music & Audio'
    app_name = r'com_clearchannel_iheartradio_controller'
    serial_app_dir = os.path.join(SERIAL_DIR, category, app_name)
    u_days = map(get_update_day, os.listdir(r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew\{}'.format(app_name)))
    features, days, chess_data = get_chess_data_content(os.path.join(serial_app_dir, r'UC_Serial.xlsx'))
    y_label = 'Central Features'
    svg_path = os.path.join(serial_app_dir, r'UC_Serial.svg')
    draw_chessboard(chess_data, plt.cm.gray_r, days, filter_update_days(list(u_days)), features, y_label, app_name,
                    svg_path)
    '''
    '''
    for category,app_name in [['Travel & Local', 'com_yelp_android'],['Music & Audio', 'com_smule_singandroid']]:
        serial_app_dir = os.path.join(SERIAL_DIR, category, app_name)
        u_days = map(get_update_day, os.listdir(
            r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew\{}'.format(app_name)))
        features, days, chess_data = get_chess_data_content(os.path.join(serial_app_dir, r'UC_Serial.xlsx'))
        y_label = 'Central features'
        svg_path = os.path.join(serial_app_dir, r'UC_Serial.svg')
        draw_chessboard(chess_data, plt.cm.gray_r, days, filter_update_days(list(u_days)), features, y_label, app_name,
                        svg_path)
    '''

    category = 'Music & Audio'
    app_name = r'com_smule_singandroid'
    serial_app_dir = os.path.join(SERIAL_DIR, category, app_name)
    u_days = map(get_update_day, os.listdir(
        r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew\{}'.format(app_name)))
    features, days, chess_data = get_chess_data_content(os.path.join(serial_app_dir, r'UC_Serial.xlsx'))
    y_label = 'Central features'
    svg_path = os.path.join(serial_app_dir, r'UC_Serial.svg')
    draw_chessboard(chess_data, plt.cm.gray_r, days, filter_update_days(list(u_days)), features, y_label, app_name,
                    svg_path)

    '''
    for category, app_name in [['Travel & Local', 'com_yelp_android'], ['Music & Audio', 'com_smule_singandroid']]:
        serial_app_dir = os.path.join(SERIAL_DIR, category, app_name)
        u_days = map(get_update_day, os.listdir(
            r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew\{}'.format(app_name)))
        days, chess_data = get_chess_data_time(os.path.join(serial_app_dir, r'UT_Serial.xlsx'))
        svg_path = os.path.join(serial_app_dir, r'UT_Serial.svg')
        draw_chessboard(chess_data, plt.cm.gray, days, filter_update_days(list(u_days)), '', '', app_name, svg_path)
    '''

    category = 'Travel & Local'
    app_name = r'com_yelp_android'
    serial_app_dir = os.path.join(SERIAL_DIR, category, app_name)
    u_days = map(get_update_day, os.listdir(
        r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew\{}'.format(app_name)))
    days, chess_data = get_chess_data_time(os.path.join(serial_app_dir, r'UT_Serial.xlsx'))
    svg_path = os.path.join(serial_app_dir, r'UT_Serial.svg')
    draw_chessboard(chess_data, plt.cm.gray, days, filter_update_days(list(u_days)), '', '', app_name, svg_path)

