# -*- coding:utf-8 -*-
'''
Created on 2019.03.07

@author: Molly Song
'''


import os
import math
import pandas
from sklearn.svm import LinearSVC
from predict_update_content import get_data
from collections import defaultdict



CATEGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'
CORR_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'
THRESHOLD_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result\UC_Threshold'
SERIAL_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Serial_predict'
WHATSNEW_DIR = r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew'
MIN_PERIOD = 2
MIN_TRAIN_DATA_COUNT = 50


def choose_start_day(category, app, valid_udate_days):
    train_data_count = 0
    num_update_days = len(valid_udate_days)
    for i in range(num_update_days - 1):
        train_data_count += (valid_udate_days[i+1] - valid_udate_days[i] - MIN_PERIOD)
        if train_data_count > MIN_TRAIN_DATA_COUNT:
            break

    if train_data_count < MIN_TRAIN_DATA_COUNT:
        return -1, -1

    start_day = valid_udate_days[max(i, math.ceil(num_update_days/2))] + MIN_PERIOD

    app_data_dir = os.path.join(CORR_DIR, category, app, 'Update_content')
    df_day = pandas.read_excel(os.path.join(app_data_dir, os.listdir(app_data_dir)[0], 'dates_by_day.xlsx'))
    find_result = df_day[df_day.day == start_day]
    if find_result.empty:
        return -1, -1
    day_index = int(find_result.id)
    return start_day, day_index


def predict_update_content_successively(category, app, start_index, step):
    threshold_path = os.path.join(THRESHOLD_DIR, '{}.xlsx'.format(app))
    if not os.path.exists(threshold_path):
        return

    df_thresholds = pandas.read_excel(threshold_path)
    app_data_dir = os.path.join(CORR_DIR, category, app, 'Update_content')
    df_day = pandas.read_excel(os.path.join(app_data_dir, os.listdir(app_data_dir)[0], 'dates_by_day.xlsx'))
    df_dict = defaultdict(list)

    # count = 0
    for feature in os.listdir(app_data_dir):
        print('\t%s' % feature)
        feature_dir = os.path.join(app_data_dir, feature)

        threshold = df_thresholds[df_thresholds.feature == feature].threshold
        if threshold.empty:
            continue
        threshold = float(threshold)

        x, y = get_data(feature_dir, threshold)
        if len(x) > 0:
            df_dict['feature'].append(feature)
        one_round_index = start_index
        while one_round_index < len(x):
            # count += 1
            day = int(df_day[df_day.id == one_round_index].day)
            if len(set(y[:one_round_index])) == 1:
                df_dict[day].append(y[0])
            else:
                clf = LinearSVC()
                clf.fit(x[:one_round_index], y[:one_round_index])
                y_predicted = clf.predict([x[one_round_index]])
                df_dict[day].extend(y_predicted)
            one_round_index += step

    # print(count)
    # for k, v in df_dict.items():
    #     print('{}:{}'.format(k, len(v)))

    serial_category_path = os.path.join(SERIAL_DIR, category, app)
    if not os.path.exists(serial_category_path):
        os.makedirs(serial_category_path)
    excel_writer = pandas.ExcelWriter(os.path.join(serial_category_path, 'UC_Serial.xlsx'))
    df = pandas.DataFrame(data=df_dict)
    df.to_excel(excel_writer)
    excel_writer.save()


def get_update_day(file_name):
    return int(file_name.split('.')[0])


def filter_update_days(days, start_day):
    num_days = len(days)
    days.sort()
    return [days[i] for i in range(num_days) if days[i] >= start_day-MIN_PERIOD and (i - 1 < 0 or days[i] - days[i - 1] > MIN_PERIOD)]


if __name__ == '__main__':
    '''
    category = 'Music & Audio'
    app_name = 'com_clearchannel_iheartradio_controller'
    start_index = 197  # date:248
    step = 1
    predict_update_content_successively(category, app_name, start_index, step)
    
    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    start_day = 133
    start_index = 86
    # step = 1
    # predict_update_content_successively(category, app_name, start_index, step)
    valid_update_days = filter_update_days(list(u_days), start_day)
    evaluate_stability(app_name, valid_update_days)
    '''

    df_dic = defaultdict(list)
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    # for category in ['Music & Audio', 'Photography', 'Personalization',
    #                  'Productivity', 'Tools', 'Weather', 'Lifestyle']:
    # for category in ['Entertainment', 'Games']:
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Entertainment', 'Games']:
    #     file_name = '{}.txt'.format(category)
    for file_name in os.listdir(CATEGORY_PATH):
        category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            u_days = map(get_update_day, os.listdir(os.path.join(WHATSNEW_DIR, app_name)))
            valid_update_days = filter_update_days(list(u_days), 0)
            start_day, start_index = choose_start_day(category, app_name, valid_update_days)

            if start_index == -1:
                continue

            df_dic['App'].append(app_name)
            df_dic['start_day'].append(start_day)
            step = 1
            predict_update_content_successively(category, app_name, start_index, step)

    excel_writer = pandas.ExcelWriter(os.path.join(SERIAL_DIR, 'Serial_start_day.xlsx'))
    df = pandas.DataFrame(data=df_dic)
    df.to_excel(excel_writer)
    excel_writer.save()



