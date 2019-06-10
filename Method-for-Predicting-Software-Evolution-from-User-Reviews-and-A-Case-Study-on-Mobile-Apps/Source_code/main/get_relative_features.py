# -*- coding:utf-8 -*-
'''
Created on 2019.01.16

@author: MollySong
'''


import os
import pandas


def get_update_times(feature_path, present_time):
    f = open(feature_path)
    lines = f.readlines()
    f.close()
    times = []
    for line in lines:
        time, flag = map(int, line.strip('\n').split(':::'))
        if time > present_time:
            break
        if flag:
            times.append(time)
    return times


SLOT_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


def get_window(app_name):
    df_lag = pandas.read_excel(os.path.join(SLOT_DIR, 'User_reaction_lag.xlsx'))
    window = int(2 * df_lag[df_lag.App == app_name].average_slot / 3)
    return window


def is_relative_feature(app1, feature_dir1, feature_dir2, one_feature, time):
    update_times1 = get_update_times(os.path.join(feature_dir1, one_feature,
                                                  'WhatsNew.txt'), time)
    update_times2 = get_update_times(os.path.join(feature_dir2, one_feature,
                                                  'WhatsNew.txt'), time)

    if len(update_times1) < 3 or len(update_times2) < 3:
        return False

    window = get_window(app1)
    update_times2 = set(update_times2)
    follow_count = 0.0
    for ut in update_times1:
        if set(range(ut - window, ut + window + 1)) & update_times2:
            follow_count += 1.0

    if follow_count / len(update_times1) > 0.6:
        return True


PSOURCE_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataPredictable'


def find_relative_features(category1, app1, category_app_list, time):
    feature_dir1 = os.path.join(PSOURCE_DIR, category1, app1)
    features1 = os.listdir(feature_dir1)
    relative_features = []
    for category2, app2 in category_app_list:
        feature_dir2 = os.path.join(PSOURCE_DIR, category2, app2)
        features2 = os.listdir(feature_dir2)
        common_feature = set(features1) & set(features2)
        for one_feature in common_feature:
            if is_relative_feature(app1, feature_dir1, feature_dir2, one_feature, time):
                relative_features.append(one_feature)
    return relative_features


# GetAllArray
def whether_follow_update(category1, app1, category_app_list, time, aim_feature):
    whether = 0
    feature_dir1 = os.path.join(PSOURCE_DIR, category1, app1)
    for category2, app2 in category_app_list:
        feature_dir2 = os.path.join(PSOURCE_DIR, category2, app2)
        if not os.path.exists(feature_dir2):
            return whether

        features = os.listdir(feature_dir2)
        if aim_feature not in features:
            continue
        one_whether = is_relative_feature(app1, feature_dir1, feature_dir2, aim_feature, time)
        whether = whether or one_whether
        if whether:
            return whether
    return whether


if __name__ == '__main__':
    pass