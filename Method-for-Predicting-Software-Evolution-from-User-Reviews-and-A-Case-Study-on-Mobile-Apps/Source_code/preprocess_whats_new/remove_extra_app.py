# -*- coding:utf-8 -*-
'''
Created on 2018.11.15

@author: MollySong
'''

import os
import shutil


def remove_app_data(aim_dir):
    for app_name in os.listdir(aim_dir):
        if not os.path.exists(os.path.join(STD_DIR, app_name)):
            print(app_name)
            shutil.rmtree(os.path.join(aim_dir, app_name))


STD_DIR = 'D:\programming\workspacePycharm\masterProject\show_app\WhatsNew'
if __name__ == '__main__':
    aim_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2017_03_29'
    remove_app_data(aim_dir)
