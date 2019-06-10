# -*- coding:utf-8 -*-
'''
Created on 2019.04.06

@author: MollySong
'''


import os
import numpy
import pandas
from collections import defaultdict
CATEGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'
APPDATA_DIR = r'D:\programming\workspacePycharm\masterProject\show_app\AppData'
RATING_DIR = r'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\Daily_rating'
LAG_PATH = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data\User_reaction_lag.xlsx'
STABLE_DAY_PATH = r'D:\programming\workspacePycharm\masterProject\Data\Serial_predict\UC_min_day.xlsx'
PREDICTION_RESULT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'
DATA_DIR = r'D:\programming\workspacePycharm\masterProject\Data'


def get_app_avg_daily_review_number(category, app):
    folder_name = app.replace('_', '.')
    df_path = os.path.join(APPDATA_DIR, category, folder_name, 'review_information.xlsx')
    if not os.path.exists(df_path):
        return None
    df = pandas.read_excel(df_path)
    return float(df[['review numbers']].mean())


def get_app_avg_daily_rating(category, app):
    df_path = os.path.join(RATING_DIR, category, '{}.xlsx'.format(app))
    if not os.path.exists(df_path):
        return None
    df = pandas.read_excel(df_path)
    return float(df[['rating']].mean())


def get_reaction_lag_and_avg_release_slot(app):
    df = pandas.read_excel(LAG_PATH)
    find = df[df.App == app]
    if find.empty:
        return None
    return int(find.lag), float(find.average_slot)


def get_serial_content_stable_day(app):
    df = pandas.read_excel(STABLE_DAY_PATH)
    find = df[df.App == app]
    if find.empty:
        return None
    return int(find.Min_day)


def get_avg_content_prediction_accuracy(category, app):
    f_path = os.path.join(PREDICTION_RESULT_DIR, category, app, 'UC_accuracy.txt')
    if not os.path.exists(f_path):
        return None
    f = open(f_path, 'r')
    lines = f.readlines()
    f.close()
    accuracies = list(map(lambda x: float(x.strip('\n')), lines))
    if not accuracies:
        return None
    return numpy.mean(accuracies)


def get_avg_time_prediction_accuracy(category, app):
    f_path = os.path.join(PREDICTION_RESULT_DIR, category, app, 'UT.txt')
    if not os.path.exists(f_path):
        return None
    f = open(f_path, 'r')
    line = f.readline()
    f.close()
    return float(line.split(' ')[-1])


def get_avg_satisfaction_prediction_accuracy(category, app):
    f_path = os.path.join(PREDICTION_RESULT_DIR, category, app, 'US.txt')
    if not os.path.exists(f_path):
        return 0, 0, 0
    f = open(f_path, 'r')
    line = f.readlines()[-1]
    f.close()
    return list(map(float, line.split(' ')))


if __name__ == '__main__':
    df_data = defaultdict(list)
    for file_name in os.listdir(CATEGORY_PATH):
        category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            df_data['Category'].append(category)
            df_data['App'].append(app_name)

            adrn = get_app_avg_daily_review_number(category, app_name)
            df_data['Avg_review_number'].append(adrn)

            adr = get_app_avg_daily_rating(category, app_name)
            df_data['Avg_rating'].append(adr)

            rl, s = get_reaction_lag_and_avg_release_slot(app_name)
            df_data['Reaction_lag'].append(rl)
            df_data['Avg_release_slot'].append(s)

            csd = get_serial_content_stable_day(app_name)
            df_data['Serial_stable_day'].append(csd)

            cpa = get_avg_content_prediction_accuracy(category, app_name)
            df_data['Content_prediction_accuracy'].append(cpa)

            tpa = get_avg_time_prediction_accuracy(category, app_name)
            df_data['Time_prediction_accuracy'].append(tpa)

            ipa, ppa, npa = get_avg_satisfaction_prediction_accuracy(category, app_name)
            if ipa == 0:
                df_data['Intensity_prediction_accuracy'].append(None)
            else:
                df_data['Intensity_prediction_accuracy'].append(ipa)
            if ppa == 0:
                df_data['Pog_sentiment_prediction_accuracy'].append(None)
            else:
                df_data['Pog_sentiment_prediction_accuracy'].append(ppa)
            if npa == 0:
                df_data['Neg_sentiment_prediction_accuracy'].append(None)
            else:
                df_data['Neg_sentiment_prediction_accuracy'].append(npa)

    excel_writer = pandas.ExcelWriter(os.path.join(DATA_DIR, 'App_integrated_data.xlsx'))
    df = pandas.DataFrame(data=df_data)
    df.to_excel(excel_writer)
    excel_writer.save()
