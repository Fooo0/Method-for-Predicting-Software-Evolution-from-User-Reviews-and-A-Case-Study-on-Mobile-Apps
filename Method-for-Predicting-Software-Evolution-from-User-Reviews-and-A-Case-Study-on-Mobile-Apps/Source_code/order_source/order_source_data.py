'''
Created on 2018.1.18

@author: Molly Song
'''

import re
import os
import datetime


def get_whatsnew_day_n_text(matches):
    for item in matches:
        if item:
            return item.group('days'), item.group('text')
    print("NO MATCH")
    exit(0)


def order_n_write_data(dir_path, data_list):
    data_list.sort()    # sort
    day_to_write = None
    one_day_reviews = []
    for item in data_list:
        item_day = item[0]
        item_review = item[1]
        if not day_to_write:
            day_to_write = item_day
            one_day_reviews = [item_review]
        else:
            if day_to_write == item_day:
                one_day_reviews.append(item_review)
            else:
                one_day_f = open(os.path.join(dir_path, "%d.txt" % day_to_write), 'a', encoding='utf-8')
                for r in one_day_reviews:
                    one_day_f.write("%s\n" % r)
                one_day_f.close()
                day_to_write = item_day
                one_day_reviews = [item_review]
    one_day_f = open(os.path.join(dir_path, "%d.txt" % day_to_write), 'a', encoding='utf-8')    # last day
    for r in one_day_reviews:
        one_day_f.write("%s\n" % r)
    one_day_f.close()


DAYREF = datetime.datetime(2016,3,1)
DAYREF_WH = datetime.datetime(2016,1,1)
PAT_NAME =re.compile(r'^(?P<name>[a-z]+(\.\w*(_\w*)*)*)\s\d+',re.I)    # App name
# 3 kinds of update description
PAT_DES1 = re.compile(r'^\d+\s(?P<days>\d+)\s(?P<text>.*)',re.I)
PAT_DES2 = re.compile(r'^\d+\s(?P<days>\d+)\s\d*[\.,)]\s*(?P<text>.*)',re.I)
PAT_DES3 = re.compile(r'^\d+\s(?P<days>\d+)\s\W+\s*(?P<text>.*)',re.I)
PATTERNS = [PAT_NAME, PAT_DES2, PAT_DES3, PAT_DES1]
PATH_ORDER_WH = r"D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Whatsnew"
PATH_ORDER_REV = r"D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Review"


def get_source_data(file_path, is_review):
    a_file = open(file_path, 'r', encoding='utf-8', errors='ignore')
    lines = a_file.readlines()
    lineNum = len(lines)    # length
    a_file.close()
    if is_review:    #  for review
        f_path = os.path.join(PATH_ORDER_REV, file_path.split("\\")[-1].split(".")[0])
        if not os.path.exists(f_path):
            print("make")
            os.makedirs(f_path)    # create directory
        dateLine = 0
        reviewLine = 1
        day_review_pairs = []
        while reviewLine < lineNum:
            date = datetime.datetime.strptime(lines[dateLine].replace('DATE:',''),"%B %d, %Y\n")
            day_now = (date - DAYREF).days + 1
            print(day_now)
            review = lines[reviewLine].replace('REVB:','').strip('\n')
            day_review_pairs.append([day_now, review])
            dateLine += 7
            reviewLine += 7
        order_n_write_data(f_path, day_review_pairs)
    else:    # for update information
        day_whatsnew_pairs = []
        app_to_write = None
        for line in lines:
            matches = [pattern.match(line) for pattern in PATTERNS]    # match all the patterns
            if matches[0]:    # match id for App
                app_now = matches[0].group('name').replace('.', '_')
                if app_to_write:
                    print(app_to_write)
                    f_path = os.path.join(PATH_ORDER_WH, app_to_write)
                    if not os.path.exists(f_path):
                        os.makedirs(f_path)    # create directory
                    order_n_write_data(f_path, day_whatsnew_pairs)
                app_to_write = app_now
                day_whatsnew_pairs = []
            else:
                day_now, one_inf = get_whatsnew_day_n_text(matches[1:])
                day_whatsnew_pairs.append([int(day_now)-(DAYREF - DAYREF_WH).days, one_inf])
        f_path = os.path.join(PATH_ORDER_WH, app_to_write)
        if not os.path.exists(f_path):
            os.makedirs(f_path)    # create directory
        order_n_write_data(f_path, day_whatsnew_pairs)
    return


def get_all_review_source(dir_path):
    for f_name in os.listdir(dir_path):
        get_source_data(os.path.join(dir_path, f_name), True)

get_all_review_source(r'D:\programming\workspacePycharm\masterProject\Data\OriginalSource\Googleplay_AllReview')
# print('~~~~~~~~~~~~~~~~~~~~~~~~~~~Finish Review~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~')
# get_source_data(r'D:\programming\workspacePycharm\masterProject\Data\OriginalSource\Whatsnew.txt', False)
