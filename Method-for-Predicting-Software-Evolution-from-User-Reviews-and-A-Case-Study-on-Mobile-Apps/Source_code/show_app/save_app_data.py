# -*- coding:utf-8 -*-
'''
Created on 2018.11.12

@author: Molly Song
'''

import re
import os
import pandas
import datetime
from collections import defaultdict

# installs: <div class="content" itemprop="numDownloads">***</div>
# offfered by: 提供者：</div> <div class="content">Shifty Jelly</div>
# developer: 访问网站</a> <a class="dev-link" href="mailto:androidsupport@shiftyjelly.com"
# rating:<meta content="4.630371570587158" itemprop="ratingValue">

# REVCOUNTPAT = re.compile('<span class="reviews-num" aria-label=".+>39,737 (?P<aim>\d+(,*\d+)*)</span>', re.I)

INSTALLPAT = re.compile(
    '<div class="content" itemprop="numDownloads">\s*(?P<aim>\d+(,*\d+)*( - \d+(,*\d+)*)?)\s*</div>', re.I)
OFFERPAT = re.compile('[(提供者：\s*)(\s*Offered By\s*)]\s*</div>\s*<div class="content">\s*(?P<aim>[^<]+)\s*</div>', re.I)
DEVELOPERPAT = re.compile('href="mailto\s*:\s*(?P<aim>[^"]+)"', re.I)
SCOREPAT = re.compile('<meta content="(?P<aim>\d+(\.\d+)?)" itemprop="ratingValue">', re.I)
TIMEPAT = re.compile('(?P<aim>\d+).txt', re.I)
DAYREF = datetime.datetime(2016, 3, 1)
DAYREF_WH = datetime.datetime(2016, 1, 1)
GAPDAY = (DAYREF - DAYREF_WH).days


def get_match_aim(match_pattern, one_str):
    search_result = match_pattern.search(one_str)
    if not search_result:
        return '0'
    return search_result.group('aim')


def record_dict_in_file(record_dict, file_path):
    f = open(file_path, 'w', encoding='utf-8')
    for k, vs in record_dict.items():
        f.write('%s\n' % k)
        for v in vs:
            f.write('\t%s\n' % v)
        f.write('\n')
    f.close()


if __name__ == '__main__':
    category_dir = 'D:\programming\workspacePycharm\masterProject\AppCategory'
    source_whats_new_dir = 'D:\programming\workspacePycharm\masterProject\show_app\WhatsNew'
    source_review_dir = 'D:\programming\workspacePycharm\masterProject\Data\OrderedSource\Review'
    app_data_dir = 'D:\programming\workspacePycharm\masterProject\show_app\AppData'
    offered_by_dict = defaultdict(set)
    developer_dict = defaultdict(set)
    for cate_txt in os.listdir(category_dir):
        category = cate_txt.split('.')[0]
        print(category)

        source_category_file = open(os.path.join(category_dir, cate_txt), 'r')
        app_name_lines = source_category_file.readlines()
        source_category_file.close()

        for app_name_line in app_name_lines:
            # one app information from what's new
            app_name_line = app_name_line.strip('\n')
            app_id = app_name_line.replace('_', '.')
            print(app_id)

            source_app_wdir = os.path.join(source_whats_new_dir, app_id)
            app_update_time = []
            app_installs = []
            app_offeredby = []
            app_developer = []
            app_score = []
            time_pattern = re.compile('%s(?P<aim>\d+).txt' % app_name_line, re.I)
            if not os.path.exists(source_app_wdir):
                continue
            for one_whats_new in os.listdir(source_app_wdir):
                relative_time = int(get_match_aim(time_pattern, one_whats_new)) - GAPDAY
                app_wfile = open(os.path.join(source_app_wdir, one_whats_new), 'r', encoding='utf-8', errors='ignore')
                html_lines = app_wfile.readlines()
                app_wfile.close()

                install_num = get_match_aim(INSTALLPAT, html_lines[1])
                offered_by = get_match_aim(OFFERPAT, html_lines[1])
                developer = get_match_aim(DEVELOPERPAT, html_lines[1])
                score = get_match_aim(SCOREPAT, html_lines[3])

                app_update_time.append(relative_time)
                app_installs.append(install_num)
                app_offeredby.append(offered_by)
                app_developer.append(developer)
                app_score.append(score)

                offered_by_dict[offered_by].add('<' + category + '>' + app_id)
                developer_dict[developer].add('<' + category + '>' + app_id)

            if len(app_installs) <= 0:  # no what's new information
                continue

            one_category_path = os.path.join(app_data_dir, category)
            if not os.path.exists(one_category_path):
                os.mkdir(one_category_path)
            one_app_path = os.path.join(one_category_path, app_id)
            if not os.path.exists(one_app_path):
                os.mkdir(one_app_path)

            excel_writer1 = pandas.ExcelWriter(os.path.join(one_app_path, 'whatsnew_information.xlsx'))
            df1 = pandas.DataFrame(data={'update_time': app_update_time, 'installs': app_installs, 'score': app_score,
                                         'offered_by': app_offeredby, 'developer': app_developer, })
            df1.to_excel(excel_writer1, index=False)
            excel_writer1.save()

            # one app information from user review
            source_app_rdir = os.path.join(source_review_dir, app_name_line)
            days_of_review = []
            num_of_reviews = []
            last_day = -1
            for daily_review in sorted(os.listdir(source_app_rdir), key=lambda x: int(x.split('.')[0])):
                day = int(daily_review.split('.')[0])
                print(day)
                while last_day > 0 and (day - last_day) != 1:
                    last_day += 1
                    days_of_review.append(last_day)
                    num_of_reviews.append(0)
                app_rfile = open(os.path.join(source_app_rdir, daily_review), 'r')
                num = len(app_rfile.readlines())
                app_rfile.close()

                days_of_review.append(day)
                num_of_reviews.append(num)
                last_day = day

            pairs = list(zip(days_of_review, num_of_reviews))
            pairs.sort()
            days_of_review[:], num_of_reviews[:] = zip(*pairs)

            excel_writer2 = pandas.ExcelWriter(os.path.join(one_app_path, 'review_information.xlsx'))
            df2 = pandas.DataFrame(data={'day': days_of_review, 'review numbers': num_of_reviews, })
            df2.to_excel(excel_writer2, index=False)
            excel_writer2.save()

    record_dict_in_file(offered_by_dict, os.path.join(app_data_dir, 'offered by.txt'))
    record_dict_in_file(developer_dict, os.path.join(app_data_dir, 'developer.txt'))

