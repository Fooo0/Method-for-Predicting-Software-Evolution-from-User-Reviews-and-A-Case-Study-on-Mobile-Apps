# -*- coding:utf-8 -*-
'''
Created on 2019.01.17

@author: Molly Song
'''


import os
import numpy


def show_review_data_distribution(review_dir):
    max_daily_num = 0
    app_review_nums = []
    for app in os.listdir(review_dir):
        all_num = 0
        app_dir = os.path.join(review_dir, app)
        for day in os.listdir(app_dir):
            f = open(os.path.join(app_dir, day), 'r')
            daily_num = len(f.readlines())
            f.close()
            if daily_num > max_daily_num:
                max_daily_num = daily_num
            all_num += daily_num
        app_review_nums.append(all_num)

    print(max_daily_num)
    print(sum(app_review_nums))
    print(min(app_review_nums))
    print(max(app_review_nums))
    print(numpy.mean(app_review_nums))


def show_whats_new_data_distribution(whats_new_dir):
    app_update_times = []
    for app in os.listdir(whats_new_dir):
        update_times = len(os.listdir(os.path.join(whats_new_dir, app)))
        app_update_times.append(update_times)

    print(sum(app_update_times))
    print(min(app_update_times))
    print(max(app_update_times))
    print(numpy.mean(app_update_times))


PATH_ORDER = r'D:\programming\workspacePycharm\masterProject\Data\OrderedSource'


if __name__ == '__main__':
    show_review_data_distribution(os.path.join(PATH_ORDER, 'Review'))
    show_whats_new_data_distribution(os.path.join(PATH_ORDER, 'Whatsnew'))
