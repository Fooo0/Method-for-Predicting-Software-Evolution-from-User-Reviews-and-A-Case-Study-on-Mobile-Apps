# -*- coding:utf-8 -*-
'''
Created on 2018.11.22

@author: MollySong
'''

import os
import sys
import numpy
import pandas
import openpyxl
from collections import Counter
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split


def predict_update_time(x, y):
    clf = GaussianNB()
    scores = cross_val_score(clf, numpy.matrix(x), numpy.array(y), scoring='accuracy', cv=5)
    return scores


def get_data(data_dir, threshold):
    filter_file_path = os.path.join(data_dir, 'feature_kendall.xlsx')
    vaild_features_path = os.path.join(data_dir, 'valid_features.xlsx')
    wb = openpyxl.load_workbook(filter_file_path)
    sheet_names = wb.sheetnames
    # print(sheet_names)
    x = []
    filtered_feature_count = 0
    first_row = True
    for sn in sheet_names:
        # print('Now process sheet%s' % sn)
        df_kendall = pandas.read_excel(filter_file_path, sheet_name=sn)
        df_filtered_kendall = df_kendall[abs(df_kendall.kendall_correlation_coefficient) > threshold]
        filtered_feature_names = df_filtered_kendall.index.tolist()
        filtered_feature_count += len(filtered_feature_names)
        if filtered_feature_names:
            filtered_features = pandas.read_excel(vaild_features_path, sheet_name=sn).\
                                    loc[:, filtered_feature_names]
            row_index = 0
            for _, row in filtered_features.iterrows():
                feature_values = row.tolist()
                if first_row:
                    x.append(feature_values)
                else:
                    x[row_index].extend(feature_values)
                row_index += 1
            # if first_row:
            #     print('row_index=', row_index)
            first_row = False

    if not filtered_feature_count:
        return [], []

    df_labels = pandas.read_excel(os.path.join(data_dir, 'labels.xlsx'))
    y = df_labels['label'].tolist()

    print('\tThreshold=%f, Number of filtered features: %d' % (threshold, filtered_feature_count))
    return x, y


THRESHOLD_EFFECT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result\Threshold_effect'


def predict_with_varying_threshold(category, app_name):
    app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_time')
    avg_scores = []
    valid_thresholds = []
    threshold = 1.0
    while threshold >= -0.05:
        x, y = get_data(app_data_dir, threshold)
        if len(set(y)) <= 1:
            # print('\t\tNo correlated property can be used for prediction')
            threshold -= 0.05
            continue
        try:
            scores = predict_update_time(x, y)
        except BaseException:
            threshold -= 0.05
            continue

        avg_score = numpy.mean(scores)
        avg_scores.append(avg_score)
        valid_thresholds.append(threshold)
        threshold -= 0.05

    save_dir = os.path.join(THRESHOLD_EFFECT_DIR, app_name)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    f = open(os.path.join(save_dir, 'UT.txt'), 'w')
    for t, s in zip(valid_thresholds, avg_scores):
        f.write('{} {}\n'.format(t, s))
    f.close()



DATA_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'
PREDICT_RESULT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'


def get_classificaton_detail(category, app_name):
    predict_result_file = open(os.path.join(PREDICT_RESULT_DIR, category, app_name, 'UT.txt'), 'r')
    threshold = float(predict_result_file.readline().split(' ')[0])
    print(threshold)
    predict_result_file.close()

    app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_time')
    x, y = get_data(app_data_dir, threshold)

    all_test = []
    all_predict = []
    for j in range(5):
        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)
        clf = GaussianNB()
        clf.fit(x_train, y_train)
        y_predict = clf.predict(x_test)
        all_test.extend(y_test)
        all_predict.extend(y_predict)
    print(classification_report(all_test, all_predict))


if __name__ == '__main__':
    '''
    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    predict_with_varying_threshold(category, app_name)
    '''

    '''
    category_app_pairs = [['Music & Audio', 'com_smule_singandroid'],
                          ['Travel & Local', 'com_yelp_android'],
                          ['Social', 'com_snapchat_android'],
                          ['Photography', 'com_google_android_apps_photos'],
                          ['Shopping', 'com_amazon_mShop_android_shopping']]
    for category, app_name in category_app_pairs:
        print(app_name)
        get_classificaton_detail(category, app_name)
    '''

    category_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'
    predict_result_path = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'
    # for category in ['Books & Reference', 'Business', 'Education', 'Social', 'Communication',
    #                  'Finance', 'Maps & Navigation', 'News & Magazines', 'Travel & Local']:
    # for category in ['Music & Audio', 'Photography', 'Personalization',
    #                  'Productivity', 'Tools', 'Weather', 'Lifestyle']:
    # for category in ['Entertainment', 'Games']:
    avg_scores = []
    valid_thresholds = []
    for category in ['Travel & Local']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(category_path):
    #     category = file_name.split('.')[0]
        print(category)
        # f = open(os.path.join(category_path, file_name), 'r')
        # app_names = f.readlines()
        # f.close()
        app_names = ['com_yelp_android']
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)

            app_data_dir = os.path.join(DATA_DIR, category, app_name, 'Update_time')
            if not os.path.exists(app_data_dir):
                continue

            print('\tStart getting data')
            threshold = 1.0
            best_threshold = 0
            best_y = None
            best_scores = None
            best_avg_score = 0
            worse_count = 0
            while threshold >= -0.05:
                x, y = get_data(app_data_dir, threshold)
                if len(set(y)) <= 1:
                    threshold -= 0.05
                    continue

                try:
                    scores = predict_update_time(x, y)
                except BaseException:
                    threshold -= 0.05
                    continue

                avg_score = numpy.mean(scores)
                avg_scores.append(avg_score)
                valid_thresholds.append(threshold)
                if avg_score > best_avg_score:
                    best_threshold = threshold
                    best_y = y
                    best_scores = scores
                    best_avg_score = avg_score
                #     worse_count = 0
                # else:
                #     worse_count += 1
                # if worse_count >= 3:
                #     break
                threshold -= 0.05

            if best_scores is None:
                continue

            num_y = len(best_y)
            for k, v in Counter(best_y).items():
                print('%d : %f(%d/%d)' % (k, float(v) / num_y, v, num_y), end='\t')

            print('\n\tAll Accuracies:', end='\n\t')
            for s in best_scores:
                print('\t%s' % s)
            print('\tAverage Accuracy: %f' % best_avg_score)

            predict_dir = os.path.join(predict_result_path, category, app_name)
            if not os.path.exists(predict_dir):
                os.makedirs(predict_dir)
            predict_result_file = open(os.path.join(predict_dir, 'UT.txt'), 'w')
            predict_result_file.write('{:.2f} {}\n'.format(best_threshold, best_avg_score))
            predict_result_file.close()

            save_dir = os.path.join(THRESHOLD_EFFECT_DIR, app_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)
            f = open(os.path.join(save_dir, 'UT.txt'), 'w')
            for t, s in zip(valid_thresholds, avg_scores):
                f.write('{} {}\n'.format(t, s))
            f.close()
