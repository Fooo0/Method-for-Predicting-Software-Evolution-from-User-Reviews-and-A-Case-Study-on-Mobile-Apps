# -*- coding:utf-8 -*-
'''
Created on 2018.11.20

@author: MollySong
'''

import os
import sys
import numpy
import pandas
import openpyxl
import tsfresh
from collections import defaultdict
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table

PSOURCE_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataPredictable'
RECORD_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'
RATING_DIR = r'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\Daily_rating'


def generate_y_label(one_y):
    if one_y < 14:
        return 0
    if 14 <= one_y < 35:
        return 1
    return 2


def pick_update_time(txt_line):
    return int(txt_line.split(':::')[0])


MIN_PERIOD = 2


def generate_time_series(category, app_name):
    source_data_dir = os.path.join(PSOURCE_DIR, category, app_name)
    time_series = defaultdict(list)
    set_y = False
    y = []
    series_lens = []
    df_rating = pandas.read_excel(os.path.join(RATING_DIR, category, '%s.xlsx'%app_name))
    all_rating_series = []
    for feature in os.listdir(source_data_dir):
        print('Feature: %s' % feature)
        feature_path = os.path.join(source_data_dir, feature)
        file_whatsnew = open(os.path.join(feature_path, 'WhatsNew.txt'), 'r')  # get update time stamp
        lines_whatsnew = file_whatsnew.readlines()
        file_whatsnew.close()
        update_times = list(map(pick_update_time, lines_whatsnew))
        num_update = len(update_times)

        file_rev = open(os.path.join(feature_path, 'Review.txt'), 'r')  # file for property
        lines_rev = file_rev.readlines()
        file_rev.close()

        line_num = 0
        for i in range(num_update-1):
            if update_times[i+1]-update_times[i] <= MIN_PERIOD:
                continue
            freq_series = []
            pos_series = []
            neg_series = []
            rating_series = []
            print('Update day: %d' % update_times[i])
            while line_num < len(lines_rev):
                line_spl = lines_rev[line_num].split(':::')
                day_rev = int(line_spl[0])  # day of review
                print('Review day: %d' % day_rev)
                if day_rev < update_times[i]:  # last update
                    line_num += 1
                    continue
                if day_rev >= update_times[i+1]:
                    break

                freq_series.append(float(line_spl[1]))
                pos_series.append(float(line_spl[2]))  # positive sentiment score
                neg_series.append(float(line_spl[3].strip('\n')))  # negative sentiment score
                if not set_y:
                    rating_series.append(float(df_rating[df_rating.time == day_rev]['rating']))  # rating
                if len(freq_series) >= MIN_PERIOD+1:
                    time_series['%s_freq' % feature].extend(freq_series)
                    time_series['%s_pos' % feature].extend(pos_series)
                    time_series['%s_neg' % feature].extend(neg_series)
                    if not set_y:
                        series_lens.append(len(freq_series))
                        # y.append(float(day_rev-update_times[i])/(update_times[i+1]-day_rev))
                        y.append(update_times[i+1]-day_rev)
                        all_rating_series.extend(rating_series)

                line_num += 1  # next day
        set_y = True

    time_series['rating'] = all_rating_series
    series_id = 0
    for item in series_lens:
        time_series['time'].extend(range(item))
        time_series['id'].extend([series_id]*item)
        series_id += 1  # next id

        time_prediction_record_path = os.path.join(RECORD_DIR, category, app_name, 'Update_time')
    if not os.path.exists(time_prediction_record_path):
        os.makedirs(time_prediction_record_path)

    # print(time_series)
    excel_writer = pandas.ExcelWriter(os.path.join(time_prediction_record_path, 'time_series.xlsx'))
    df_series = pandas.DataFrame(data=time_series)
    df_series.to_excel(excel_writer)
    excel_writer.save()

    # cats = pandas.qcut(y, 3, duplicates='drop')
    # label_details = cats.categories
    # print(label_details)
    # labels = cats.codes
    labels = list(map(generate_y_label, y))

    df_label = pandas.DataFrame(data={'id': range(len(labels)), 'label': labels})
    excel_writer = pandas.ExcelWriter(os.path.join(time_prediction_record_path, 'labels.xlsx'))
    df_label.to_excel(excel_writer)
    excel_writer.save()

    # file_label_detail = open(os.path.join(app_dir, 'Label_details.txt'), 'w')
    # file_label_detail.write('%s' % label_details)
    # file_label_detail.close()
    return df_series, df_label


def filter_features(app_dir, df_series, labels):
    filtered_feature = tsfresh.extract_relevant_features(df_series, labels, column_id='id', column_sort='time', ml_task='classification')

    excel_writer = pandas.ExcelWriter(os.path.join(app_dir, 'filtered_feature.xlsx'))
    df_filtered_features = pandas.DataFrame(data=filtered_feature)
    df_filtered_features.to_excel(excel_writer)
    excel_writer.save()


def generate_valid_features(app_dir, df_series):
    column_nums = df_series.shape[1] - 2
    sheet_count = 0
    valid_features = []
    valid_feature_count = 0
    excel_writer = pandas.ExcelWriter(os.path.join(app_dir, 'valid_features.xlsx'))
    for i in range(0, column_nums, 6):
        column_range = list(range(i, min(i+6, column_nums)))
        column_range.extend([-2, -1])

        new_valid_features = extract_features(df_series.iloc[:, column_range], column_id='id',
                                              column_sort='time', show_warnings=False).\
            replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=1, how='any')
        valid_features.append(new_valid_features)

        df_new_features = pandas.DataFrame(data=new_valid_features)
        valid_feature_count += (df_new_features.shape[1] - 1)
        df_new_features.to_excel(excel_writer, '%d' % sheet_count)

        sheet_count += 1

    excel_writer.save()
    print('Number of valid features: %d' % valid_feature_count)
    return valid_features


def calculate_feature_relevance(app_dir, valid_features_list, df_labels):
    excel_writer1 = pandas.ExcelWriter(os.path.join(app_dir, 'feature_p_value.xlsx'))
    excel_writer2 = pandas.ExcelWriter(os.path.join(app_dir, 'feature_kendall.xlsx'))
    sheet_count = 0
    for df_valid_features in valid_features_list:  # one sheet of features
        # calculate p-value
        print(sheet_count)
        df_p_value = calculate_relevance_table(df_valid_features, df_labels['label'], ml_task='classification')
        df_p_value.to_excel(excel_writer1, '%d' % sheet_count)

        # arrange data format
        df_valid_features.reset_index(inplace=True)
        # print(df_valid_features.columns)
        filtered_feature = {'id', }
        for _, row in df_p_value.iterrows():
            if row['relevant']:
                filtered_feature.add(row['feature'])
        filtered_by_p = df_valid_features[list(filtered_feature)]
        # print(filtered_by_p[list(filtered_feature)[-1]])
        # print(filtered_by_p[list(filtered_feature)[-2]])
        # print(filtered_by_p['id'])
        # print(df_labels['id'])
        feature_with_label = pandas.merge(filtered_by_p, df_labels, on='id').drop('id', 1)
        # print(feature_with_label)
        # sys.exit(0)

        # calculate kendall correlation coefficient
        feature_kendall = feature_with_label.corr('kendall')['label'].drop('label', 0).to_frame()
        feature_kendall.rename(columns={'label': 'kendall_correlation_coefficient'}, inplace=True)
        df_kendall = pandas.DataFrame(feature_kendall)
        df_kendall.to_excel(excel_writer2, '%d' % sheet_count)

        sheet_count += 1
    excel_writer1.save()
    excel_writer2.save()


if __name__ == '__main__':
    '''
    category = r'Shopping'
    app_name = r'com_c51'


    category_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'
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
        f = open(os.path.join(category_path, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            print('Start generating time series and labels')
            df_series, df_labels = generate_time_series(category, app_name)

            time_prediction_record_path = os.path.join(RECORD_DIR, category, app_name, 'Update_time')

            # df_series = pandas.read_excel(os.path.join(app_dir, 'time_series.xlsx'))
            print('Start generating valid features')
            valid_features_list = generate_valid_features(time_prediction_record_path, df_series)

            print('Start calculating feature revelance')
            # vaild_feature_file_path = os.path.join(time_prediction_record_path, 'valid_features.xlsx')
            # wb = openpyxl.load_workbook(vaild_feature_file_path)
            # sheet_names = wb.sheetnames
            # valid_features_list = []
            # for sn in sheet_names:
            #     valid_features = pandas.read_excel(vaild_feature_file_path, sheet_name=sn)
            #     valid_features_list.append(valid_features)
            # df_labels = pandas.read_excel(os.path.join(time_prediction_record_path, 'labels.xlsx'))
            calculate_feature_relevance(time_prediction_record_path, valid_features_list, df_labels)
    '''

    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    df_series, df_labels = generate_time_series(category, app_name)
    time_prediction_record_path = os.path.join(RECORD_DIR, category, app_name, 'Update_time')
    print('Start generating valid features')
    valid_features_list = generate_valid_features(time_prediction_record_path, df_series)
    print('Start calculating feature revelance')
    calculate_feature_relevance(time_prediction_record_path, valid_features_list, df_labels)
