# -*- coding:utf-8 -*-
'''
Created on 2018.12.04

@author: MollySong
'''


import os
import numpy
import pandas
from collections import Counter
from sklearn.svm import LinearSVC
from sklearn.model_selection import cross_val_score, train_test_split
from find_high_rating_app import get_higher_rating_app
from get_relative_features import whether_follow_update



DATA_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


def get_data(feature_dir, threshold):
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
    for _, row in filtered_features.iterrows():
        _x = row.tolist()
        x.append(_x)

    df_labels = pandas.read_excel(os.path.join(feature_dir, 'labels.xlsx'))
    y = df_labels['label'].tolist()

    print('\t\tThreshold=%f, Number of filtered features: %d' % (threshold, filtered_feature_count))
    return x, y


def predict_one_feature(x, y):
    clf = LinearSVC()
    scores = cross_val_score(clf, numpy.matrix(x), numpy.array(y), scoring='accuracy', cv=5)
    return scores


def get_data_1(feature_dir, threshold):
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


def predict_one_feature_1(category, app_name, aim_feature, x, y):
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


def predict_one_app(category, app_name, threshold):
    scores = []
    app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_content')
    for feature in os.listdir(app_data_dir):
        print('\t%s' % feature)
        feature_dir = os.path.join(app_data_dir, feature)
        x, y = get_data(feature_dir, threshold)
        if not x:
            # print('\t\tNo correlated property can be used for prediction')
            continue

        num_y = len(y)
        for k, v in Counter(y).items():
            print('\t\t%d : %f(%d/%d)' % (k, float(v) / num_y, v, num_y), end='')
        print('')

        ss = predict_one_feature(x, y)
        scores.extend(ss)
        for s in ss:
            print('\t\t%f' % s)
        print('\t\tAverage Accuracy for One Feature: %f' % numpy.mean(ss))

    print('Average Accuracy for One App: %f\n' % numpy.mean(scores))
    return scores


THRESHOLD_EFFECT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result\Threshold_effect'


def predict_with_varying_threshold(category, app_name, feature, file_name):
    feature_dir = os.path.join(DATA_DIR, category, app_name, 'Update_content', feature)
    avg_scores = []
    valid_thresholds = []
    threshold = 1.0
    while threshold >= -0.05:
        x, y = get_data(feature_dir, threshold)
        if len(set(y)) <= 1:
            # print('\t\tNo correlated property can be used for prediction')
            threshold -= 0.05
            continue
        try:
            ss = predict_one_feature(x, y)
        except BaseException:
            threshold -= 0.05
            continue

        avg_score = numpy.mean(ss)
        avg_scores.append(avg_score)
        valid_thresholds.append(threshold)
        threshold -= 0.05

    save_dir = os.path.join(THRESHOLD_EFFECT_DIR, app_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, file_name), 'w')
    for t, s in zip(valid_thresholds, avg_scores):
        f.write('{} {}\n'.format(t, s))
    f.close()


if __name__ == '__main__':
    category = 'Music & Audio'
    app_name = 'com_smule_singandroid'
    feature = r'reset song'
    predict_with_varying_threshold(category, app_name, feature, 'UC_reset_song.txt')

    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    feature = r'autorotation annoying'
    predict_with_varying_threshold(category, app_name, feature, 'UC__autorotation_annoying.txt')
    '''

    category_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'
    predict_result_path = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'

    threshold_dir = os.path.join(predict_result_path, 'UC_Threshold')
    if not os.path.exists(threshold_dir):
        os.mkdir(threshold_dir)

    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    # for category in ['Music & Audio', 'Photography', 'Personalization',
    #                  'Productivity', 'Tools', 'Weather', 'Lifestyle']:
    # for category in ['Entertainment', 'Games']:
    for category in ['Shopping']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(category_path):
    #     category = file_name.split('.')[0]
        print(category)
        # f = open(os.path.join(category_path, file_name), 'r')
        # app_names = f.readlines()
        # f.close()
        app_names = ['com_offerup', 'com_walmart_android']
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            scores = []
            app_features = []
            feature_thresholds = []

            app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_content')
            if not os.path.exists(app_data_dir):
                continue

            result_dir = os.path.join(predict_result_path, category, app_name)
            if not os.path.exists(result_dir):
                os.makedirs(result_dir)

            predict_result_file = open(os.path.join(result_dir, 'UC_accuracy.txt'), 'w')
            # threshold_file = open(os.path.join(result_dir, 'UC_threshold.txt'), 'w')
            for feature in os.listdir(app_data_dir):
                print('\t%s' % feature)
                feature_dir = os.path.join(app_data_dir, feature)

                threshold = 1.0
                best_threshold = 0
                best_y = None
                best_scores = None
                best_avg_score = 0
                worse_count = 0
                while threshold >= 0.1:
                    x, y = get_data(feature_dir, threshold)
                    # x, y = get_data_1(feature_dir, threshold)
                    if len(set(y)) <= 1:
                        # print('\t\tNo correlated property can be used for prediction')
                        threshold -= 0.05
                        continue
                    try:
                        ss = predict_one_feature(x, y)
                    except BaseException:
                        threshold -= 0.05
                        continue

                    # ss = predict_one_feature_1(category, app_name, feature, x, y)
                    # if not ss:
                    #     if best_avg_score:
                    #         worse_count += 1
                    #     threshold -= 0.05
                    #     continue

                    avg_score = numpy.mean(ss)
                    if avg_score > best_avg_score:
                        best_y = y
                        best_threshold = threshold
                        best_scores = ss
                        best_avg_score = avg_score
                        worse_count = 0
                    else:
                        worse_count += 1

                    if worse_count >= 3:
                        break

                    threshold -= 0.05

                if best_scores is None:    # no effective(kendall) prediction features
                    continue
                scores.extend(best_scores)
                app_features.append(feature)
                feature_thresholds.append(best_threshold)
                # threshold_file.write('{:.2f}\n'.format(best_threshold))

                num_y = len(best_y)
                for k, v in Counter(best_y).items():
                    print('\t\t%d : %f(%d/%d)' % (k, float(v) / num_y, v, num_y), end='')
                print(' ')
                for s in best_scores:
                    print('\t\t%f' % s)
                print('\t\tAverage Accuracy for One Feature: %f' % best_avg_score)
                predict_result_file.write('{}\n'.format(best_avg_score))

            # if len(scores) != 0:
            #     predict_result_file.write('{}\n'.format(numpy.mean(scores)))

            predict_result_file.close()

            excel_writer = pandas.ExcelWriter(os.path.join(threshold_dir, '{}.xlsx'.format(app_name)))
            df = pandas.DataFrame(data={'feature': app_features, 'threshold': feature_thresholds})
            df.to_excel(excel_writer, float_format='%.2f')
            excel_writer.save()
    '''