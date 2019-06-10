# -*- coding:utf-8 -*-
'''
Created on 2019.03.07

@author: Molly Song
'''


import os
import pandas
from predict_update_time import get_data
from sklearn.naive_bayes import GaussianNB


DATA_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'
PREDICT_RESULT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'
SERIAL_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Serial_predict'


if __name__ == '__main__':
    # category = 'Music & Audio'
    # app_name = 'com_smule_singandroid'
    # start_index = 179  # date:224

    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    start_index = 86  # date:133

    step = 1

    predict_result_file = open(os.path.join(PREDICT_RESULT_DIR, category, app_name, 'UT.txt'), 'r')
    threshold = float(predict_result_file.readline().split(' ')[0])
    print(threshold)
    predict_result_file.close()

    content_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_content')
    time_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_time')
    df_datess = pandas.read_excel(os.path.join(content_data_dir, os.listdir(content_data_dir)[0], 'dates_by_day.xlsx'))
    dates_by_day = df_datess['day'].tolist()
    print(dates_by_day)

    x, y = get_data(time_data_dir, threshold)
    days = []
    prediction_result = []
    test_index = start_index
    print('len(x)={},len(y)={}'.format(len(x), len(y)))
    while test_index < len(x):
        print(test_index)
        clf = GaussianNB()
        clf.fit(x[:test_index], y[:test_index])
        y_predicted = clf.predict([x[test_index]])

        days.append(dates_by_day[test_index])
        prediction_result.extend(y_predicted)
        test_index += step

    serial_category_path = os.path.join(SERIAL_DIR, category, app_name)
    if not os.path.exists(serial_category_path):
        os.makedirs(serial_category_path)
    excel_writer = pandas.ExcelWriter(os.path.join(serial_category_path, 'UT_Serial.xlsx'))
    df = pandas.DataFrame(data={'day': days, 'prediction': prediction_result})
    df.to_excel(excel_writer)
    excel_writer.save()

    '''

    category = 'Communication'
    app_name = 'kik_android'
    start_index = 128  # date:173

    predict_result_file = open(os.path.join(PREDICT_RESULT_DIR, category, app_name, 'UT.txt'), 'r')
    threshold = float(predict_result_file.readline().split(' ')[0])
    predict_result_file.close()

    content_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_content')
    time_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_time')
    df_datess = pandas.read_excel(os.path.join(content_data_dir, os.listdir(content_data_dir)[0], 'dates_by_day.xlsx'))
    dates_by_day = df_datess['day'].tolist()

    x, y = get_data(time_data_dir, threshold)
    days = []
    prediction_result = []
    test_index = start_index
    while test_index < len(x):
        print(test_index)
        clf = GaussianNB()
        clf.fit(x[:test_index], y[:test_index])
        y_predicted = clf.predict([x[test_index]])

        days.append(dates_by_day[test_index])
        prediction_result.extend(y_predicted)
        test_index += step

    serial_category_path = os.path.join(SERIAL_DIR, category, app_name)
    if not os.path.exists(serial_category_path):
        os.makedirs(serial_category_path)
    excel_writer = pandas.ExcelWriter(os.path.join(serial_category_path, 'UT_Serial.xlsx'))
    df = pandas.DataFrame(data={'day': days, 'prediction': prediction_result})
    df.to_excel(excel_writer)
    excel_writer.save()
    '''


