# -*- coding:utf-8 -*-
'''
Created on 2019.03.31

@author: Molly Song
'''

import os
import math
import numpy
import pandas
from pylab import mpl
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from matplotlib.ticker import MultipleLocator
from serial_content_prediction import choose_start_day

CATEGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'
CORR_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'
SERIAL_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Serial_predict'
WHATSNEW_DIR = r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew'
MIN_PERIOD = 2


def get_evaluation_time(app_name):
    df_lag = pandas.read_excel(os.path.join(CORR_DIR, 'User_reaction_lag.xlsx'))
    evaluation_time = int(df_lag[df_lag.App == app_name].average_slot) - MIN_PERIOD
    return evaluation_time


def calculate_entropy(l):
    all_count = len(l)
    entropy = 0
    for _, v in Counter(l).items():
        ratio = v / all_count
        entropy += (-ratio) * math.log(ratio, 2)
    return entropy


def evaluate_stability_a(category, app, valid_update_days):
    serial_app_path = os.path.join(SERIAL_DIR, category, app)
    df = pandas.read_excel(os.path.join(serial_app_path, 'UC_Serial.xlsx'))
    features = df['feature'].tolist()

    evaluation_time = get_evaluation_time(app)
    num_days = []
    avg_entropies = []
    for jump in range(evaluation_time - MIN_PERIOD + 1):
        entropies = []
        for f in features:  # all central features
            for i in range(len(valid_update_days) - 1):
                start_day = valid_update_days[i] + MIN_PERIOD
                end_day = start_day + jump
                if end_day >= valid_update_days[i + 1]:
                    continue
                predictions = list(map(int, df[df.feature == f][list(range(start_day, end_day))].values[0]))
                entropies.append(calculate_entropy(predictions))
        if not entropies:
            break
        avg_entropy = numpy.mean(entropies)
        avg_entropies.append(avg_entropy)
        num_days.append(jump + MIN_PERIOD)

    excel_writer = pandas.ExcelWriter(os.path.join(serial_app_path, 'UC_entropy_a.xlsx'))
    df_enrtopies = pandas.DataFrame(data={'End_day': num_days, 'Average_entropy': avg_entropies})
    df_enrtopies.to_excel(excel_writer)
    excel_writer.save()

    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_major_locator = MultipleLocator(5)
    x_minor_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)

    plt.title(app)
    ax.set_xlabel('Timeline:the t-th Day from latest update')
    ax.set_ylabel('Entropy')
    ax.plot(num_days, avg_entropies, '-', color='black')

    plt.savefig(os.path.join(serial_app_path, 'UC_entropy_a.svg'), format='svg')
    plt.close()


def find_first_local_minimum(l):
    index = 0
    last_value = 1
    while index < len(l):
        if l[index] > last_value:
            break
        last_value = l[index]
        index += 1
    return index - 1


def evaluate_stability_b(category, app, valid_update_days):
    serial_app_path = os.path.join(SERIAL_DIR, category, app)
    df_path = os.path.join(serial_app_path, 'UC_Serial.xlsx')
    if not os.path.exists(df_path):
        return -1
    df = pandas.read_excel(df_path)
    df_columns = df.columns.values.tolist()
    if not df_columns:
        return -1
    features = df['feature'].tolist()

    evaluation_time = get_evaluation_time(app)
    num_days = []
    avg_entropies = []
    for jump in range(evaluation_time - MIN_PERIOD + 1):
        entropies = []
        for f in features:  # all central features
            for i in range(len(valid_update_days) - 1):
                start_day = valid_update_days[i] + MIN_PERIOD + jump
                end_day = valid_update_days[i + 1]
                if start_day >= end_day - 1:
                    continue
                predictions = list(map(int, df[df.feature == f] \
                    [list(set(range(start_day, end_day)) & set(df_columns))].values[0]))
                entropies.append(calculate_entropy(predictions))
        if not entropies:
            break
        avg_entropy = numpy.mean(entropies)
        avg_entropies.append(avg_entropy)
        num_days.append(jump + MIN_PERIOD)

    excel_writer = pandas.ExcelWriter(os.path.join(serial_app_path, 'UC_entropy_b.xlsx'))
    df_enrtopies = pandas.DataFrame(data={'Start_day': num_days, 'Average_entropy': avg_entropies})
    df_enrtopies.to_excel(excel_writer)
    excel_writer.save()

    mpl.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x_major_locator = MultipleLocator(5)
    x_minor_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)

    plt.title(app)
    ax.set_xlabel('Timeline:the t-th Day from latest update')
    ax.set_ylabel('Entropy')
    ax.plot(num_days, avg_entropies, '-', color='black')

    plt.savefig(os.path.join(serial_app_path, 'UC_entropy_b.svg'), format='svg')
    plt.close()

    key_index = find_first_local_minimum(avg_entropies)
    if key_index < 0:
        return -1

    return num_days[key_index]


def get_update_day(file_name):
    return int(file_name.split('.')[0])


def filter_update_days(days, start_day):
    num_days = len(days)
    days.sort()
    if start_day != 0:
        return [days[i] for i in range(num_days) if
                days[i] >= start_day - MIN_PERIOD and days[i] - days[i - 1] > MIN_PERIOD]
    return [days[i] for i in range(num_days) if
            (i - 1 < 0) or (days[i] >= start_day - MIN_PERIOD and days[i] - days[i - 1] > MIN_PERIOD)]


if __name__ == '__main__':
    '''
    category = 'Music & Audio'
    app_name = 'com_clearchannel_iheartradio_controller'
    start_index = 197  # date:248
    step = 1
    evaluate_stability_b(category, app_name, start_index, step)
    '''
    '''
    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    # start_day = 133
    # start_index = 86
    # step = 1
    # predict_update_content_successively(category, app_name, start_index, step)
    df_start_day = pandas.read_excel(os.path.join(SERIAL_DIR, 'Serial_start_day.xlsx'))
    find_row = df_start_day[df_start_day.App == app_name]
    start_day = int(find_row.start_day)
    u_days = map(get_update_day, os.listdir(os.path.join(WHATSNEW_DIR, app_name)))
    valid_update_days = filter_update_days(list(u_days), start_day)
    # evaluate_stability_a(category, app_name, valid_update_days)
    evaluate_stability_b(category, app_name, valid_update_days)
    '''

    df_dic = defaultdict(list)
    # for category in ['Entertainment', 'Games']:
        # file_name = '{}.txt'.format(category)
    for file_name in os.listdir(CATEGORY_PATH):
        category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            u_days = list(map(get_update_day, os.listdir(os.path.join(WHATSNEW_DIR, app_name))))

            # valid_update_days = filter_update_days(u_days, 0)
            # start_day, start_index = choose_start_day(category, app_name, valid_update_days)
            # if start_index == -1:
            #     continue

            df_start_day = pandas.read_excel(os.path.join(SERIAL_DIR, 'Serial_start_day.xlsx'))
            find_row = df_start_day[df_start_day.App == app_name]
            if find_row.empty:
                continue
            start_day = int(find_row.start_day)
            valid_update_days = filter_update_days(u_days, start_day)
            # print(start_day)
            # print(valid_update_days)

            min_d = evaluate_stability_b(category, app_name, valid_update_days)
            if min_d > 0:
                df_dic['App'].append(app_name)
                df_dic['Min_day'].append(min_d)

    excel_writer = pandas.ExcelWriter(os.path.join(SERIAL_DIR, 'UC_min_day.xlsx'))
    df_min_d = pandas.DataFrame(data=df_dic)
    df_min_d.to_excel(excel_writer)
    excel_writer.save()

