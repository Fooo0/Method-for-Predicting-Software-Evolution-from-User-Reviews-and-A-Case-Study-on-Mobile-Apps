# -*- coding:utf-8 -*-
'''
Created on 2018.12.04

@author: MollySong
'''


import os
import numpy
import pandas
from collections import defaultdict
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table


PSOURCE_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataPredictable'
RECORD_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


def get_update_information(line):
    return tuple(map(int, line.strip().split(':::')))


MIN_PERIOD = 2


def generate_time_series(srcdir, desdir):
    file_whatsnew = open(os.path.join(srcdir, 'WhatsNew.txt'), 'r')  # get update time stamp
    lines_whatsnew = file_whatsnew.readlines()
    file_whatsnew.close()
    update_times = list(map(get_update_information, lines_whatsnew))
    num_update = len(update_times)

    file_rev = open(os.path.join(srcdir, 'Review.txt'), 'r')  # file for property
    lines_rev = file_rev.readlines()
    file_rev.close()

    line_num = 0
    time_series = defaultdict(list)
    labels = []
    series_lens = []
    dates_by_day = []
    for i in range(num_update-1):
        if update_times[i+1][0] - update_times[i][0] <= MIN_PERIOD:
            continue
        freq_series = []
        pos_series = []
        neg_series = []
        # print('Update day: %d' % update_times[i][0])
        while line_num < len(lines_rev):
            line_spl = lines_rev[line_num].split(':::')
            day_rev = int(line_spl[0])  # day of review
            # print('Review day: %d' % day_rev)
            if day_rev < update_times[i][0]:  # last update
                line_num += 1
                continue
            if day_rev >= update_times[i + 1][0]:
                break

            freq_series.append(float(line_spl[1]))
            pos_series.append(float(line_spl[2]))  # positive sentiment score
            neg_series.append(float(line_spl[3].strip('\n')))  # negative sentiment score
            if len(freq_series) >= MIN_PERIOD+1:
                time_series['frequency'].extend(freq_series)
                time_series['positive_sentiment'].extend(pos_series)
                time_series['negative_sentiment'].extend(neg_series)
                labels.append(update_times[i + 1][1])
                series_lens.append(len(freq_series))
                dates_by_day.append(day_rev)

            line_num += 1  # next day

    series_id = 0
    for item in series_lens:
        time_series['time'].extend(range(item))
        time_series['id'].extend([series_id] * item)
        series_id += 1  # next id

    excel_writer = pandas.ExcelWriter(os.path.join(desdir, 'time_series.xlsx'))
    df_series = pandas.DataFrame(data=time_series)
    df_series.to_excel(excel_writer)
    excel_writer.save()

    excel_writer = pandas.ExcelWriter(os.path.join(desdir, 'labels.xlsx'))
    df_label = pandas.DataFrame(data={'id': range(len(labels)), 'label': labels})
    df_label.to_excel(excel_writer)
    excel_writer.save()

    excel_writer = pandas.ExcelWriter(os.path.join(desdir, 'dates_by_day.xlsx'))
    dates_label = pandas.DataFrame(data={'id': range(len(labels)), 'day': dates_by_day})
    dates_label.to_excel(excel_writer)
    excel_writer.save()

    return df_series, df_label


def generate_valid_features(desdir, df_series):
    valid_features = extract_features(df_series, column_id='id', column_sort='time', show_warnings=False).\
        replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=1, how='any')
    excel_writer = pandas.ExcelWriter(os.path.join(desdir, 'valid_features.xlsx'))
    df_valid_features = pandas.DataFrame(data=valid_features)
    df_valid_features.to_excel(excel_writer)
    excel_writer.save()

    print('Number of valid features: %d' % (df_valid_features.shape[1] - 1))
    return valid_features


def calculate_feature_relevance(desdir, valid_features, df_labels):
    # calculate p-value
    excel_writer1 = pandas.ExcelWriter(os.path.join(desdir, 'feature_p_value.xlsx'))
    try:
        df_p_value = calculate_relevance_table(valid_features, df_labels['label'], ml_task='classification')
    except BaseException:
        return
    df_p_value.to_excel(excel_writer1)
    excel_writer1.save()

    # arrange data format
    valid_features.reset_index(inplace=True)
    filtered_feature = {'id', }
    for _, row in df_p_value.iterrows():
        if row['relevant']:
            filtered_feature.add(row['feature'])
    if len(filtered_feature) <= 1:  # No correlated feature
        return
    filtered_by_p = valid_features.loc[:, list(filtered_feature)]
    feature_with_label = pandas.merge(filtered_by_p, df_labels, on='id').drop('id', 1)

    # calculate kendall correlation coefficient
    feature_kendall = feature_with_label.corr('kendall')['label'].drop('label', 0).to_frame()
    feature_kendall.rename(columns={'label': 'kendall_correlation_coefficient'}, inplace=True)
    excel_writer2 = pandas.ExcelWriter(os.path.join(desdir, 'feature_kendall.xlsx'))
    df_kendall = pandas.DataFrame(feature_kendall)
    df_kendall.to_excel(excel_writer2)
    excel_writer2.save()


def analyse_one_feature(srcdir, desdir):
    print('Start generating time series')
    df_series, df_labels = generate_time_series(srcdir, desdir)

    print('Start generating valid features')
    valid_features = generate_valid_features(desdir, df_series)

    print('Start calculating feature revelance')
    calculate_feature_relevance(desdir, valid_features, df_labels)


def analyse_one_app(category, app_name):
    source_data_dir = os.path.join(PSOURCE_DIR, category, app_name)
    record_data_dir = os.path.join(RECORD_DIR, category, app_name, 'Update_content')
    for feature in os.listdir(source_data_dir):
        print(feature)
        desdir = os.path.join(record_data_dir, feature)
        if not os.path.exists(desdir):
            os.makedirs(desdir)
        analyse_one_feature(os.path.join(source_data_dir, feature), desdir)


if __name__ == '__main__':
    category_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    # for category in ['Music & Audio', 'Photography', 'Personalization',
    #                  'Productivity', 'Tools', 'Weather', 'Lifestyle', 'Shopping']:
    # for category in ['Entertainment', 'Games']:
    for category in ['Communication', 'Finance', 'Maps & Navigation',
                     'News & Magazines', 'Travel & Local']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(category_path):
    #     category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(category_path, file_name), 'r')
        app_names = f.readlines()
        f.close()
        if category == 'Communication':
            app_names = ['kik_android']
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            analyse_one_app(category, app_name)
    '''
    category = 'Travel & Local'
    app_name = 'com_yelp_android'
    analyse_one_app(category, app_name)
    '''
