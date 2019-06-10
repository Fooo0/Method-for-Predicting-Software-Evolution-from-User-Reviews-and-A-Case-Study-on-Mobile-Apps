# -*- coding:utf-8 -*-
'''
Created on 2018.11.29

@author: MollySong
'''


import os
import re
import numpy
import pandas
import datetime
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib import cm
from collections import Counter
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator



CATAGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'
WHATSNEW_DIR = r'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\WhatsNew'
RATING_DIR = r'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\Daily_rating_O'
RECORD_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'
EXTENDING = 5
DAYREF = datetime.datetime(2016, 3, 1)
DAYREF_WH = datetime.datetime(2016, 1, 1)
GAPDAY = (DAYREF - DAYREF_WH).days


def get_daily_rating(category, app_name):
    app_rating_path = os.path.join(os.path.join(RATING_DIR, category), '%s.xlsx' % app_name)
    df_rating = pandas.read_excel(app_rating_path)
    df_rating.set_index('time', inplace=True)
    return df_rating


def get_update_day(match_pattern, one_str):
    search_result = match_pattern.search(one_str)
    if not search_result:
        return '0'
    return search_result.group('aim')


def remove_successive_day(days):
    num_days = len(days)
    days.sort()
    return [days[i] for i in range(num_days) if i-1 < 0 or i+1 >= num_days or days[i]-days[i-1] > 1]


def get_update_time(app_name):
    _app_name = app_name.replace('_', '.')
    app_path = os.path.join(WHATSNEW_DIR, _app_name)
    time_pattern = re.compile('%s(?P<aim>\d+).txt' % app_name, re.I)
    days = []
    for file_name in os.listdir(app_path):
        update_day = int(get_update_day(time_pattern, file_name)) - GAPDAY
        days.append(update_day)
    days = remove_successive_day(days)
    df_update = pandas.DataFrame(data={'time': days, 'whether_update': [1]*len(days)})
    df_update.set_index('time', inplace=True)
    return df_update, days


def calculate_lag_pearson(df_lagged_rating, df_update):
    df_merge = pandas.merge(df_lagged_rating, df_update, how='left', on='time')
    df_merge['whether_update'].fillna(0, inplace=True)
    return df_merge.corr().values[0][1]


def get_time_slots(days):
    return [days[i] - days[i - 1] for i in range(1, len(days) - 1)]


def analyse_lag_by_app(category, app_name):
    df_rating = get_daily_rating(category, app_name)
    df_update, update_days = get_update_time(app_name)

    # fig = plt.figure()
    # ax = fig.add_subplot(111)
    # x = df_rating['rating'].index.tolist()
    # y = df_rating['rating'].values
    # ax.plot(x, y, color='#fc824a')
    # for ud in update_days:
    #     ax.axvline(ud, linestyle='--', color='#87CEFA')
    # ax.set_xlabel('Timeline:the t-th Day from 2016.3.1')
    # ax.set_ylabel('Rating')
    # app_dir = os.path.join(os.path.join(RECORD_DIR, category), app_name)
    # satisfactory_prediction_record_path = os.path.join(app_dir, 'Update_satisfactory')
    # if not os.path.exists(satisfactory_prediction_record_path):
    #     os.makedirs(satisfactory_prediction_record_path)
    # plt.savefig(os.path.join(satisfactory_prediction_record_path, 'Rating_UpdateTime.png'),
    #             dpi=200, quality=95)

    update_slots = get_time_slots(update_days)
    cleaned_slots = list(set(update_slots))
    cleaned_slots.sort()
    shortest_slot = cleaned_slots[0] if cleaned_slots[0] > 1 else cleaned_slots[1]
    pearsons = []
    for lag in range(shortest_slot + 1):
        pearsons.append(calculate_lag_pearson(df_rating.shift(lag), df_update))

    pearsons_abs = list(map(abs, pearsons))
    max_kendall_lag = pearsons_abs.index(max(pearsons_abs))
    user_reaction_lag = max_kendall_lag
    if max_kendall_lag == shortest_slot:
        max_lag = shortest_slot+EXTENDING
        for lag in range(shortest_slot + 1, max_lag + 1):
            pearsons_abs.append(abs(calculate_lag_pearson(df_rating.shift(lag), df_update)))
        max_kendall_lag = pearsons_abs.index(max(pearsons_abs))
        if max_kendall_lag < max_lag:
            user_reaction_lag = max_kendall_lag
    # print(pearsons_abs)
    # print(user_reaction_lag)
    return user_reaction_lag, numpy.mean(update_slots)


def calculate_update_slot_std(app_name):
    _, update_days = get_update_time(app_name)
    update_slots = get_time_slots(update_days)
    std = numpy.std(update_slots, ddof=1)
    return std


'''
if __name__ == '__main__':
    apps = []
    lags = []
    shortest_slots = []
    mean_slots = []
    for filename in os.listdir(CATAGORY_PATH):
        category = filename.split('.')[0]
        print(category)
        f = open(os.path.join(CATAGORY_PATH, filename))
        lines = f.readlines()
        f.close()
        for app_name in lines:
            app_name = app_name.strip()
            print(app_name)
            lag, mean_slot = analyse_lag_by_app(category, app_name)
            apps.append(app_name)
            lags.append(lag)
            mean_slots.append(mean_slot)

    # excel_writer = pandas.ExcelWriter(os.path.join(RECORD_DIR, 'User_reaction_lag.xlsx'))
    # df_lag = pandas.DataFrame(data={'App': apps, 'lag': lags, 'average_slot': mean_slots})
    # df_lag.to_excel(excel_writer)
    # excel_writer.save()

    lag_count = Counter(lags)
    labels = []
    x = []
    for k, v in sorted(lag_count.items()):
        labels.append(k)
        x.append(v)

    mpl.rcParams['font.sans-serif'] = ['Arial']
    colors = [cm.Spectral(2*0.065)]
    colors.extend(cm.Spectral(numpy.arange(3, len(lag_count)+2) * 0.055))
    # colors = ['#a552e6', '#c071fe', '#ca9bf7',
    #           '#ff796c', '#ef4026', '#e50000',
    #           '#fb7d07', '#ffa756', '#fdaa48',
    #           '#c0fa8b', '#8ee53f', '#76cd26',
    #           '#0485d1', '#069af3', '#7bc8f6',]
    fig, axes = plt.subplots(figsize=(10, 5), ncols=2)  # 设置绘图区域大小
    ax1, ax2 = axes.ravel()
    patches, texts, autotexts = ax1.pie(x, labels=labels, autopct='%1.1f%%', shadow=False, colors=colors)
    ax1.axis('equal')
    proptease = font_manager.FontProperties()
    proptease.set_size('x-small')
    plt.setp(autotexts, fontproperties=proptease)
    plt.setp(texts, fontproperties=proptease)
    # ax1.set_title('用户反映延迟', loc='center', fontproperties="SimHei")
    ax1.set_title('Lags', loc='center')
    ax2.axis('off')
    ax2.legend(patches, labels, loc='center left', fontsize='small')
    plt.tight_layout()
    plt.savefig(os.path.join(RECORD_DIR, 'User_reaction_lag.svg'), format='svg')
    # plt.savefig(os.path.join(RECORD_DIR, 'User_reaction_lag_ch.svg'), format='svg')
    # plt.close()
    '''

'''
    lag, mean_slot = analyse_lag_by_app('Games', 'com_ea_gp_minions')
    print(lag)
    print(mean_slot)
'''

if __name__ == '__main__':
    std_list = []
    for filename in os.listdir(CATAGORY_PATH):
        category = filename.split('.')[0]
        print(category)
        f = open(os.path.join(CATAGORY_PATH, filename))
        lines = f.readlines()
        f.close()
        for app_name in lines:
            app_name = app_name.strip()
            print(app_name)
            one_std = calculate_update_slot_std(app_name)
            std_list.append(one_std)

    print(min(std_list))
    print(max(std_list))

    count_dic = Counter(std_list)
    x = []
    y = []
    for k, v in count_dic.items():
        x.append(k)
        y.append(v)

    mpl.rcParams['font.sans-serif'] = ['STXihei']
    fig = plt.figure()
    fig.subplots_adjust(bottom=0.4, top=0.5)

    ax = fig.add_subplot(211)
    x_major_locator = MultipleLocator(10)
    x_minor_locator = MultipleLocator(1)
    ax.xaxis.set_major_locator(x_major_locator)
    ax.xaxis.set_minor_locator(x_minor_locator)
    ax.bar(x, y, color='black')
    plt.xticks(fontsize=8)
    plt.yticks([1], fontsize=8)
    plt.xlim(0, 160)
    plt.xlabel('更新间隔的标准差')
    plt.ylabel('数量')

    plt.savefig(os.path.join(RECORD_DIR, 'Update_slot_std_distribution.svg'), format='svg')
    plt.close()




