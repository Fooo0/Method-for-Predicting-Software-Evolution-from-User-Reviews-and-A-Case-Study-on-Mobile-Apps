# -*- coding:utf-8 -*-
'''
Created on 2018.11.27
@author: Molly Song
'''


import os
import re
import pandas
import datetime


DAYREF = datetime.datetime(2016, 3, 1)
DAYREF_WH = datetime.datetime(2016, 1, 1)
GAPDAY = (DAYREF - DAYREF_WH).days
TIMEPAT = re.compile('(?P<day>\d+).txt', re.I)
RATINGPAT = re.compile('<meta content="(?P<aim>\d+(\.\d+)?)" itemprop="ratingValue">', re.I)


def get_match_aim(match_pattern, one_str):
    search_result = match_pattern.search(one_str)
    if not search_result:
        return 0.0
    return float(search_result.group('aim'))


if __name__ == '__main__':
    category_dir = 'D:\programming\workspacePycharm\masterProject\AppCategory'
    source_whats_new_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\\2017_03_29'
    daily_rating_data_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\Daily_rating_O'

    # for cate_txt in ['Games.txt']:
    for cate_txt in os.listdir(category_dir):
        category = cate_txt.split('.')[0]
        print(category)

        source_category_file = open(os.path.join(category_dir, cate_txt), 'r')
        app_name_lines = source_category_file.readlines()
        source_category_file.close()

        # app_name_lines = ['com_ea_gp_minions']
        for app_name_line in app_name_lines:
            # one app information from what's new
            app_name_line = app_name_line.strip('\n')
            app_id = app_name_line.replace('_', '.')
            print('\t{}'.format(app_id))

            source_app_wdir = os.path.join(source_whats_new_dir, app_id)
            times = []
            last_not_zero_rating = -1
            ratings = []
            time_pattern = re.compile('%s(?P<aim>\d+).txt' % app_name_line, re.I)
            last_rtime = -1
            for one_whats_new in sorted(os.listdir(source_app_wdir), key=lambda x: int(TIMEPAT.search(x).group('day'))):
                relative_time = int(get_match_aim(time_pattern, one_whats_new)) - GAPDAY
                while last_rtime > 0 and relative_time-last_rtime > 1:
                    last_rtime += 1
                    times.append(last_rtime)
                    ratings.append(ratings[-1])

                app_wfile = open(os.path.join(source_app_wdir, one_whats_new), 'r', encoding='utf-8', errors='ignore')
                html_lines = app_wfile.readlines()
                app_wfile.close()

                times.append(relative_time)
                rating = get_match_aim(RATINGPAT, html_lines[3])
                # if rating > 0:
                #     last_not_zero_rating = rating
                # else:
                #     print('\t\t{}'.format(relative_time))
                #     rating = last_not_zero_rating
                ratings.append(rating)

                one_category_path = os.path.join(daily_rating_data_dir, category)
                if not os.path.exists(one_category_path):
                    os.mkdir(one_category_path)

                excel_writer = pandas.ExcelWriter(os.path.join(one_category_path, '%s.xlsx'%app_name_line))
                df = pandas.DataFrame(
                    data={'time': times, 'rating': ratings})
                df.to_excel(excel_writer, index=False)
                excel_writer.save()

                last_rtime = relative_time