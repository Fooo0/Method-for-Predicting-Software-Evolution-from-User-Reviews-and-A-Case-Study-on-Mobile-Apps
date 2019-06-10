# -*- coding:utf-8 -*-
'''
Created on 2017.12.11

@author: Molly Song
'''


import os
import pandas
import numpy
from collections import defaultdict
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_selection.relevance import calculate_relevance_table


PSOURCE_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataForPrediction'
RECORD_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


def pick_update_time(txt_line):
    return int(txt_line.split(':::')[0])


def get_observation_time(app_name):
    df_lag = pandas.read_excel(os.path.join(RECORD_DIR, 'User_reaction_lag.xlsx'))
    observation_time = int(df_lag[df_lag.App == app_name].average_slot / 2)
    return observation_time


def calculate_reaction_time(app_name, update_times):
    df_lag = pandas.read_excel(os.path.join(RECORD_DIR, 'User_reaction_lag.xlsx'))
    reaction_lag = int(df_lag[df_lag.App == app_name].lag)
    reaction_time = list(map(lambda x: x + reaction_lag, update_times))
    return reaction_time


def generate_y_label1(y):
    point_1 = numpy.percentile(y, 25)
    point_2 = numpy.percentile(y, 75)

    def pick_label(one_y):
        if one_y <= point_1:
            return 0
        if one_y <= point_2:
            return 1
        return 2

    labels = list(map(pick_label, y))
    return labels, [point_1, point_2]


def generate_y_label2(y):
    _y = [i for i in y if i != 0]
    if not _y:
        return [0]*len(y), [0, 0]
    point = numpy.percentile(_y, 50)
    if point > 0:
        point_1 = 0
        point_2 = point
    else:
        point_1 = point
        point_2 = 0

    def pick_label(one_y):
        if one_y <= point_1:
            return 0
        if one_y <= point_2:
            return 1
        return 2

    labels = list(map(pick_label, y))
    return labels, [point_1, point_2]


MIN_PERIOD = 2


def generate_time_series(category, app_name):
    source_data_dir = os.path.join(PSOURCE_DIR, category, app_name)
    time_series = defaultdict(list)
    series_id = 0
    update_times = None
    reaction_times = None
    num_update = None
    y_freq = []
    y_pos = []
    y_neg = []
    observation_time = get_observation_time(app_name)
    for feature in os.listdir(source_data_dir):
        print('Feature: %s' % feature)
        feature_path = os.path.join(source_data_dir, feature)
        if not update_times:
            file_whatsnew = open(os.path.join(feature_path, 'WhatsNew.txt'), 'r')  # get update time stamp
            lines_whatsnew = file_whatsnew.readlines()
            file_whatsnew.close()
            update_times = list(map(pick_update_time, lines_whatsnew))
            num_update = len(update_times)
            reaction_times = calculate_reaction_time(app_name, update_times)

        file_rev = open(os.path.join(feature_path, 'Review.txt'), 'r')  # file for property
        lines_rev = file_rev.readlines()
        file_rev.close()

        min_day = int(lines_rev[0].split(':::')[0])
        max_day = int(lines_rev[-1].split(':::')[0])
        for i in range(1, num_update):
            if update_times[i]-update_times[i-1] <= MIN_PERIOD:
                continue
            print('Update day: %d' % update_times[i])

            last_update_time = update_times[i-1]
            update_time = update_times[i]
            reaction_time = reaction_times[i]
            if update_time-observation_time < min_day \
                    or reaction_time > max_day \
                    or reaction_time+observation_time > max_day:
                continue

            index1 = last_update_time - min_day
            index2 = update_time - min_day
            index3 = reaction_time - min_day
            index4 = reaction_time + observation_time - min_day
            satisfactory_series_bf = list(map(lambda x: x.strip('\n').split(':::')[1:],
                                              lines_rev[index1:index2+1]))
            satisfactory_series_af = list(map(lambda x: x.strip('\n').split(':::')[1:],
                                              lines_rev[index3:index4+1]))

            time_series['freq'].extend([float(item[0]) for item in satisfactory_series_bf])
            time_series['pos'].extend([float(item[1]) for item in satisfactory_series_bf])
            time_series['neg'].extend([float(item[2]) for item in satisfactory_series_bf])
            time_series['time'].extend(range(len(satisfactory_series_bf)))
            time_series['id'].extend([series_id] * len(satisfactory_series_bf))
            series_id += 1

            y_freq.append(numpy.median([float(item[0]) for item in satisfactory_series_af]))
            y_pos.append(numpy.median([float(item[1]) for item in satisfactory_series_af]))
            y_neg.append(numpy.median([float(item[2]) for item in satisfactory_series_af]))

    satisfactory_prediction_record_path = os.path.join(RECORD_DIR, category, app_name, 'Update_satisfactory')
    if not os.path.exists(satisfactory_prediction_record_path):
        os.makedirs(satisfactory_prediction_record_path)

    # print(time_series)

    excel_writer = pandas.ExcelWriter(os.path.join(satisfactory_prediction_record_path, 'time_series.xlsx'))
    df_series = pandas.DataFrame(data=time_series)
    df_series.to_excel(excel_writer)
    excel_writer.save()

    df_labels = []
    postfixs = []
    label_detail_file = open(os.path.join(satisfactory_prediction_record_path, 'label_detail.txt'), 'a')

    labels, details = generate_y_label1(y_freq)
    label_detail_file.write('Frequency labels: %s \n' % ' '.join(map(str, details)))
    if len(set(labels)) > 1:
        df_label_freq = pandas.DataFrame(data={'id': range(len(labels)), 'label': labels})
        df_labels.append(df_label_freq)
        postfixs.append('freq')
        excel_writer = pandas.ExcelWriter(os.path.join(satisfactory_prediction_record_path, 'labels_freq.xlsx'))
        df_label_freq.to_excel(excel_writer)
        excel_writer.save()

    labels, details = generate_y_label2(y_pos)
    label_detail_file.write('Positive sentiment score labels: %s \n' % ' '.join(map(str, details)))
    if len(set(labels)) > 1:
        df_label_pos = pandas.DataFrame(data={'id': range(len(labels)), 'label': labels})
        df_labels.append(df_label_pos)
        postfixs.append('pos')
        excel_writer = pandas.ExcelWriter(os.path.join(satisfactory_prediction_record_path, 'labels_pos.xlsx'))
        df_label_pos.to_excel(excel_writer)
        excel_writer.save()

    labels, details = generate_y_label2(y_neg)
    label_detail_file.write('Negative sentiment score labels: %s \n' % ' '.join(map(str, details)))
    if len(set(labels)) > 1:
        df_label_neg = pandas.DataFrame(data={'id': range(len(labels)), 'label': labels})
        df_labels.append(df_label_neg)
        postfixs.append('neg')
        excel_writer = pandas.ExcelWriter(os.path.join(satisfactory_prediction_record_path, 'labels_neg.xlsx'))
        df_label_neg.to_excel(excel_writer)
        excel_writer.save()

    label_detail_file.close()

    return df_series, df_labels, postfixs


def generate_valid_features(record_dir, df_series):
    valid_features = extract_features(df_series, column_id='id', column_sort='time', show_warnings=False). \
        replace([numpy.inf, -numpy.inf], numpy.nan).dropna(axis=1, how='any')
    excel_writer = pandas.ExcelWriter(os.path.join(record_dir, 'valid_features.xlsx'))
    df_valid_features = pandas.DataFrame(data=valid_features)
    df_valid_features.to_excel(excel_writer)
    excel_writer.save()

    print('Number of valid features: %d' % (df_valid_features.shape[1] - 1))
    return valid_features


def calculate_feature_relevance(app_dir, valid_features, df_labels, postfixs):
    for df_label, postfix in zip(df_labels, postfixs):
        # calculate p-value
        excel_writer1 = pandas.ExcelWriter(os.path.join(app_dir, 'feature_p_value_%s.xlsx' % postfix))
        try:
            df_p_value = calculate_relevance_table(valid_features, df_label['label'], ml_task='classification')
        except BaseException:
            print('EXCETPTION')
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
            print('NO')
            return
        filtered_by_p = valid_features.loc[:, list(filtered_feature)]
        feature_with_label = pandas.merge(filtered_by_p, df_label, on='id').drop('id', 1)

        # calculate kendall correlation coefficient
        feature_kendall = feature_with_label.corr('kendall')['label'].drop('label', 0).to_frame()
        feature_kendall.rename(columns={'label': 'kendall_correlation_coefficient'}, inplace=True)
        excel_writer2 = pandas.ExcelWriter(os.path.join(app_dir, 'feature_kendall_%s.xlsx' % postfix))
        df_kendall = pandas.DataFrame(feature_kendall)
        df_kendall.to_excel(excel_writer2)
        excel_writer2.save()


if __name__ == '__main__':
    '''
    category = r'Shopping'
    app_name = r'com_amazon_mShop_android_shopping'
    '''
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
            df_series, df_labels, postfixs = generate_time_series(category, app_name)

            record_dir = os.path.join(RECORD_DIR, category, app_name, 'Update_satisfactory')

            print('Start generating valid features')
            valid_features = generate_valid_features(record_dir, df_series)

            print('Start calculating feature revelance(x{})'.format(len(postfixs)))
            calculate_feature_relevance(record_dir, valid_features, df_labels, postfixs)
