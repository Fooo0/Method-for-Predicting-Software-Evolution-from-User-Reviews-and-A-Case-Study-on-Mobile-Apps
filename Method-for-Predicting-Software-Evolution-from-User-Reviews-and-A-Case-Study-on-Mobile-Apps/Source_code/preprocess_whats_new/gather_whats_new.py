# -*- coding:utf-8 -*-
'''
Created on 2018.11.16

@author: MollySong
'''

import os
import datetime
import shutil

DAYREF_WH = datetime.datetime(2016, 1, 1)


def rename_whats_new(app_dir):
    for app_name in os.listdir(app_dir):
        print(app_name)
        whats_new_dir = os.path.join(app_dir, app_name)
        for daily_whats_new in os.listdir(whats_new_dir):
            file_path = os.path.join(whats_new_dir, daily_whats_new)
            day = datetime.datetime.fromtimestamp(os.path.getmtime(file_path))
            delta_day = (day - DAYREF_WH).days + 1
            app_name = app_name.replace('.', '_')
            os.rename(file_path, os.path.join(whats_new_dir, '%s%d.txt' % (app_name, delta_day)))


if __name__ == '__main__':
    aim_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2017_03_29'
    # rename_whats_new(aim_dir)

    source_dirs = ['D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2016_05_18',
                   'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2016_07_18',
                   'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2016_10_20', ]

    for app_name in os.listdir(aim_dir):
        print(app_name)
        for source_dir in source_dirs:
            shutil.copytree(os.path.join(source_dir, app_name), os.path.join(aim_dir, app_name))
