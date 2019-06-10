# -*- coding:utf-8 -*-
'''
Created on 2018.11.13

@author: MollySong
'''

import os
import re
import pandas
import matplotlib.pyplot as plt

NUMPAT = re.compile('(?P<aim>\d+(,*\d+)*)', re.I)


def get_int_install(str_install):
    search_result = NUMPAT.search(str_install)
    num_list = search_result.group('aim').split(',')
    times = len(num_list) - 1
    int_install = 0
    for item in num_list:
        added = int(item) * 10 ** (3 * times)
        int_install += added
        times -= 1
    return int_install


def get_total_review_nums(daily_review_nums):
    day_num = len(daily_review_nums)
    total_review_nums = []
    for i in range(day_num):
        total_review_nums.append(sum(daily_review_nums[:i]))
    return total_review_nums


if __name__ == '__main__':
    app_data_dir = 'D:\programming\workspacePycharm\masterProject\show_app\AppData'
    app_image_dir = 'D:\programming\workspacePycharm\masterProject\show_app\AppImage'
    for root, folders, files in os.walk(app_data_dir):
        if root != app_data_dir and len(files) == 0:
            category_name = root.split('\\')[-1]
            image_cate_dir = os.path.join(app_image_dir, category_name)
            if not os.path.exists(image_cate_dir):
                os.mkdir(image_cate_dir)
            for app_name in folders:
                print(app_name)
                data_dir = os.path.join(root, app_name)
                df1 = pandas.read_excel(os.path.join(data_dir, 'whatsnew_information.xlsx'))
                update_time = df1['update time'].values.tolist()
                scores = df1['score'].values.tolist()
                installs_str = df1['installs'].values.tolist()
                print(installs_str)

                installs_int = list(map(get_int_install, installs_str))
                print(installs_int)

                df2 = pandas.read_excel(os.path.join(data_dir, 'review_information.xlsx'))
                review_days = df2['day'].values.tolist()
                review_nums = df2['review numbers'].values.tolist()
                #                 total_review_nums = get_total_review_nums(review_nums)

                fig = plt.figure()
                fig.subplots_adjust(left=0.25, bottom=0.1, right=0.92, top=0.95)

                ax1 = fig.add_subplot(211)
                ax1.set_ylabel('Install Number', fontsize=6)
                plt.title(app_name, fontsize=7)
                plt.xlim(min(review_days), max(review_days))
                max_review_day = max(review_days)
                x_ticket = update_time + [max_review_day]
                plt.xticks(x_ticket, fontsize=5, rotation=45)
                plt.yticks(installs_int, installs_str, fontsize=6)
                ax1.bar(update_time, installs_int, width=2, color='#87CEFA', align='center')

                ax2 = ax1.twinx()
                ax2.set_ylabel('Daily Review Number', fontsize=6)
                plt.yticks(fontsize=7)
                plt.ylim(0, max(review_nums))
                ax2.plot(review_days, review_nums, color='#f36198', linewidth=0.7)

                ax3 = fig.add_subplot(212)
                base = min(list(map(int, scores)))
                ax3.set_ylabel('(Score-%d)*1000', fontsize=6)
                ax3.set_xlabel('Timeline:the t-th Day from 2016.3.1', fontsize=6)
                plt.xlim(min(review_days), max_review_day)
                plt.xticks(x_ticket, fontsize=5, rotation=45)
                plt.yticks(fontsize=7)
                clear_scores = [(i - base) * 1000 for i in scores]
                ax3.bar(update_time, clear_scores, width=2, color='#fc824a', align='center')
                pairs = list(zip(update_time, clear_scores))
                pairs.sort()
                update_time[:], clear_scores[:] = zip(*pairs)
                ax3.plot(update_time, clear_scores, ':', color='#fc824a', linewidth=0.4)

                plt.savefig(os.path.join(image_cate_dir, app_name + '.svg'), format='svg')
                plt.close()