# -*- coding:utf-8 -*-
'''
Created on 2018.11.15

@author: MollySong
'''

import os
import re
import shutil
import datetime

DATEPAT = re.compile(
    '<div class="content" itemprop="datePublished">\s*(?P<month>[a-z]+)[^\d]*(?P<day>\d+)[^\d]*(?P<year>\d+)\s*</div>',
    re.I)
TIMEPAT = re.compile('(?P<day>\d+).txt', re.I)
DAYREF_WH = datetime.datetime(2016, 1, 1)
MONTHDIC = {'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
            'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12}


def get_latest_update_time(file_line):
    search_result = DATEPAT.search(file_line)
    if not search_result:
        return -1, -1, -1
    return int(search_result.group('year')), MONTHDIC[search_result.group('month')], int(search_result.group('day'))


def get_whats_new_day1(file_name):
    search_result = TIMEPAT.search(file_name)
    return int(search_result.group('day'))


def filter_whats_new(app_dir):
    for app_name in os.listdir(app_dir):
        print(app_name)
        whats_new_dir = os.path.join(app_dir, app_name)
        #         daily_whats_new_str = app_name + r'(?P<day>\d+).txt'
        #         daily_whats_new_pat = re.compile(daily_whats_new_str, re.I)
        last_update = -1
        for daily_whats_new in sorted(os.listdir(whats_new_dir), key=get_whats_new_day1):
            file_path = os.path.join(whats_new_dir, daily_whats_new)
            update_file = open(file_path, 'r', encoding='utf-8', errors='ignore')
            update_line = update_file.readlines()[1]
            update_file.close()
            update_year, update_month, update_day = get_latest_update_time(update_line)
            if update_year == -1:  # 编码问题无法识别
                print(daily_whats_new)
                os.remove(file_path)
                continue
            latest_update_day = (datetime.datetime(update_year, update_month, update_day) - DAYREF_WH).days + 1
            if latest_update_day == last_update:
                os.remove(file_path)
            else:
                last_update = latest_update_day


## ----------------------------------------------------------------------------------------------------------------------------------------------------

PAT_NAME = re.compile(r'^(?P<name>[a-z]+(\.\w*(_\w*)*)*)\s\d+', re.I)  # App name
# 3 kinds of update description
PAT_DES1 = re.compile(r'^\d+\s(?P<days>\d+)\s.*', re.I)
PAT_DES2 = re.compile(r'^\d+\s(?P<days>\d+)\s\d*[\.,)]\s*.*', re.I)
PAT_DES3 = re.compile(r'^\d+\s(?P<days>\d+)\s\W+\s*.*', re.I)
PATTERNS = [PAT_NAME, PAT_DES2, PAT_DES3, PAT_DES1]


def get_whats_new_day2(matches):
    for item in matches:
        if item:
            return item.group('days')
    print("NO MATCH")
    exit(0)


def pick_out_whats_new(txt_path, src_dir, des_dir):
    txt_file = open(txt_path, 'r', encoding='utf-8', errors='ignore')
    lines = txt_file.readlines()
    app_name = ''
    for line in lines:
        matches = [pattern.match(line) for pattern in PATTERNS]  # match all the patterns
        if matches[0]:  # match id for App
            app_name = matches[0].group('name')
            _app_name = app_name.replace('.', '_')
        else:
            day_now = get_whats_new_day2(matches[1:])
            print(day_now)
            src_app_dir = os.path.join(src_dir, app_name)
            print(src_dir)
            if os.path.exists(src_app_dir):
                des_app_dir = os.path.join(des_dir, app_name)
                if not os.path.exists(des_app_dir):
                    os.mkdir(des_app_dir)
                shutil.copyfile(os.path.join(src_app_dir, '%s%s.txt' % (_app_name, day_now)),
                                os.path.join(des_app_dir, '%s%s.txt' % (_app_name, day_now)))


if __name__ == '__main__':
    #     app_dir1 = 'D:\programming\workspaceEclipse\MasterProject\preprocess_whats_new\\2017_03_29'
    #     filter_whats_new(app_dir1)
    txt_path = 'D:\programming\workspacePycharm\masterProject\Data\OriginalSource\Whatsnew.txt'
    src_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2017_03_29'
    des_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\WhatsNew'
    pick_out_whats_new(txt_path, src_dir, des_dir)
