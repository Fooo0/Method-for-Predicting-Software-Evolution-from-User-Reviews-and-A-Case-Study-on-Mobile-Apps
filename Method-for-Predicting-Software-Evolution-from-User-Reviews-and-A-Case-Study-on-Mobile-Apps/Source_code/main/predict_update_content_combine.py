# -*- coding:utf-8 -*-
'''
Created on 2018.03.31

@author: MollySong
'''


import os
import numpy
import pandas
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split
from find_high_rating_app import get_higher_rating_app
from get_relative_features import whether_follow_update


DATA_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


def get_data(feature_dir, threshold):
    df_dates = pandas.read_excel(os.path.join(feature_dir, 'dates_by_day.xlsx'))
    dates_by_day = df_dates['day'].tolist()

    filter_file_path = os.path.join(feature_dir, 'feature_kendall.xlsx')
    if not os.path.exists(filter_file_path):
        return [], []
    vaild_features_path = os.path.join(feature_dir, 'valid_features.xlsx')
    df_kendall = pandas.read_excel(filter_file_path)
    df_filtered_kendall = df_kendall[abs(df_kendall.kendall_correlation_coefficient) > threshold]
    filtered_feature_names = df_filtered_kendall.index.tolist()
    if not filtered_feature_names:
        return [], []
    filtered_feature_count = len(filtered_feature_names)
    x = []
    filtered_features = pandas.read_excel(vaild_features_path)[filtered_feature_names]
    index = 0
    for _, row in filtered_features.iterrows():
        _x = row.tolist()
        _x.append(dates_by_day[index])
        x.append(_x)
        index += 1

    df_labels = pandas.read_excel(os.path.join(feature_dir, 'labels.xlsx'))
    y = df_labels['label'].tolist()

    print('\t\tThreshold=%f, Number of filtered features: %d' % (threshold, filtered_feature_count))
    return x, y


def predict_one_feature(category, app_name, aim_feature, x, y):
    scores = []
    for i in range(5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)

        retry_count = 3
        while len(set(y_train)) <= 1 and retry_count > 0:
            x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.8)
            retry_count -= 1
        if len(set(y_train)) <= 1:
            continue

        x_train = [_x[:-1] for _x in x_train]
        x_test_outter = [_x[-1] for _x in x_test]
        x_test = [_x[:-1] for _x in x_test]

        clf = LinearSVC()
        clf.fit(x_train, y_train)
        y_clf = clf.predict(x_test)

        test_num = len(y_test)
        for j in range(test_num):
            if y_clf[j] == 1:
                continue
            else:
                date_time = x_test_outter[j]
                better_apps = get_higher_rating_app(app_name, date_time)
                whether = whether_follow_update(category, app_name, better_apps, date_time, aim_feature)
                if whether == 1:
                    y_clf[j] = 1

        equal_count = 0.0
        for y_act, y_pre in zip(y_test, y_clf):
            if y_act == y_pre:
                equal_count += 1
        scores.append(equal_count / test_num)

        # y_outter = []
        # for date_time in x_test_outter:
        #     better_apps = get_higher_rating_app(app_name, date_time)
        #     whether = whether_follow_update(category, app_name, better_apps, date_time, aim_feature)
        #     y_outter.append(whether)

        # y_combine = [y1 or y2 for y1, y2 in zip(y_clf, y_outter)]
        # equal_count = 0.0
        # for y_act, y_pre in zip(y_test, y_combine):
        #     if y_act == y_pre:
        #         equal_count += 1
        scores.append(equal_count / len(y_test))
    return scores


if __name__ == '__main__':
    '''
    category = r'Music & Audio'
    app_name = r'com_clearchannel_iheartradio_controller'
    feature = r'power optimising'
    predict_with_varying_threshold(category, app_name, feature, 'UC_power_optimising.txt')
    '''

    category_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'
    predict_result_path = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'

    threshold_dir = os.path.join(predict_result_path, 'UC_Threshold')

    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    for category in ['Music & Audio', 'Photography', 'Personalization',
                     'Productivity', 'Tools', 'Weather', 'Lifestyle', 'Shopping']:
    # for category in ['Entertainment', 'Games']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(category_path):
    #     category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(category_path, file_name), 'r')
        app_names = f.readlines()
        f.close()
        # app_names = ['com_yelp_android']
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)

            threshold_path = os.path.join(threshold_dir, '{}.xlsx'.format(app_name))
            if not os.path.exists(threshold_path):
                continue
            df_thresholds = pandas.read_excel(threshold_path)

            scores = []
            app_features = []
            feature_thresholds = []

            app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_content')
            if not os.path.exists(app_data_dir):
                continue

            result_dir = os.path.join(predict_result_path, category, app_name)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            predict_result_file = open(os.path.join(result_dir, 'UC_accuracy_.txt'), 'w')
            for feature in os.listdir(app_data_dir):
                print('\t%s' % feature)

                threshold = df_thresholds[df_thresholds.feature == feature].threshold
                if threshold.empty:
                    continue
                threshold = float(threshold)

                feature_dir = os.path.join(app_data_dir, feature)
                x, y = get_data(feature_dir, threshold)
                if len(set(y)) <= 1:
                    # print('\t\tNo correlated property can be used for prediction')
                    continue

                ss = predict_one_feature(category, app_name, feature, x, y)
                if not ss:
                    continue

                avg_score = numpy.mean(ss)

                scores.append(avg_score)
                app_features.append(feature)

                predict_result_file.write('{}\n'.format(avg_score))

            # if len(scores) != 0:
            #     predict_result_file.write('{}\n'.format(numpy.mean(scores)))

            predict_result_file.close()
