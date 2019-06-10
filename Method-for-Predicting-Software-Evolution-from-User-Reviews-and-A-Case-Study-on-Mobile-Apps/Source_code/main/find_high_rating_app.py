# -*- coding:utf-8 -*-
'''
Created on 2018.12.25

@author: Molly Song
'''


import os
import numpy
import pandas
from pylab import mpl
import matplotlib.pyplot as plt
from collections import defaultdict


CATAGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'
KEYWORDS_PATH = r"D:\programming\workspacePycharm\masterProject\Data\Keywords"


def get_all_app():
    app_category_dic = {}
    for file_name in os.listdir(CATAGORY_PATH):
        category = file_name.split('.')[0]
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
        category_file = open(os.path.join(CATAGORY_PATH, '{}.txt'.format(category)), 'r')
        lines = category_file.readlines()
        category_file.close()
        for line in lines:
            app_category_dic[line.strip('\n')] = category
    return app_category_dic


def get_update_days(app_name):
    file_names = os.listdir(os.path.join(KEYWORDS_PATH, 'Whatsnew', app_name))
    return list(map(lambda x: int(x.split('.')[0]), file_names))


def remove_successive_day(days):
    num_days = len(days)
    days.sort()
    return [days[i] for i in range(num_days) if i-1 < 0 or i+1 >= num_days or days[i]-days[i-1] > 1]


RECORD_DIR = r'D:\programming\workspacePycharm\masterProject\analyse_correlation\Data'


def get_apps_lag(apps):
    df_lag = pandas.read_excel(os.path.join(RECORD_DIR, 'User_reaction_lag.xlsx'))
    app_lag_dic = {}
    for app in apps:
        app_lag_dic[app] = int(df_lag[df_lag.App == app].lag)
    return app_lag_dic


def amplify_ratings(ratings):
    minimum = min(ratings)
    return list(map(lambda x: (x-minimum)*10000, ratings))


def calculate_rating_statistic(ratings):
    return numpy.median(ratings)


def calculate_line(xs, ys):
    slope, increment = list(numpy.polyfit(xs, ys, 1))
    return slope, increment


def wholely_above(slope1, increment1, slope2, increment2, b_point1, b_point2, e_point):
    if slope2*b_point2+increment2 > slope1*b_point1+increment1 and \
            slope2 * e_point + increment2 > slope1 * e_point + increment1:
        return True
    return False


MIN_PERIOD = 2
RATING_DIR = r'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\Daily_rating'


def find_latest_react_and_release_day(aim_day, app_update_days, app_lag):
    index = 0
    latest_react_day = app_update_days[index] + app_lag
    while aim_day - latest_react_day >= MIN_PERIOD:
        index += 1
        if index >= len(app_update_days):
            break
        latest_react_day = app_update_days[index] + app_lag

    if index == 0:
        return -1, -1

    if aim_day - latest_react_day < MIN_PERIOD:
        index -= 1
        latest_react_day = app_update_days[index] + app_lag
        return latest_react_day, app_update_days[index]

    return -1, -1


INSTALL_DIR = r'D:\programming\workspacePycharm\masterProject\show_app\AppData'


def get_installs(category, app, release_day):
    df_path = os.path.join(INSTALL_DIR, category, app.replace('_', '.'), 'whatsnew_information.xlsx')
    if not os.path.exists(df_path):
        return ''
    df = pandas.read_excel(df_path)
    return df[df.update_time == release_day].installs.values[0]


SUPPORT_DIR = r'D:\programming\workspacePycharm\masterProject\get_supporting_data\Data'


def filter_app(app_category_dic, aim_app, aim_day, update_days_dic, app_lag_dic):
    better_apps = []
    aim_df = pandas.read_excel(os.path.join(RATING_DIR, app_category_dic[aim_app], '%s.xlsx' % aim_app))
    aim_day = min(aim_df.loc[:, "time"].max(), aim_day)

    latest_react_day_aim, latest_release_day_aim = find_latest_react_and_release_day(aim_day, update_days_dic[aim_app], app_lag_dic[aim_app])
    if latest_react_day_aim < 0:
        return []

    install_aim = get_installs(app_category_dic[aim_app], aim_app, latest_release_day_aim)

    aim_ratings = aim_df.loc[(aim_df.time >= latest_react_day_aim) & (aim_df.time <= aim_day)].rating.tolist()
    # am_aim_ratings = amplify_ratings(aim_ratings)
    aim_median = calculate_rating_statistic(aim_ratings)
    aim_slope, aim_increment = calculate_line(range(latest_react_day_aim, aim_day + 1), aim_ratings)

    for app, category in app_category_dic.items():
    # for app, category in [['com_ea_gp_minions', 'Games']]:
        app_df = pandas.read_excel(os.path.join(RATING_DIR, category, '%s.xlsx' % app))
        aim_day_adjust = min(app_df.loc[:, "time"].max(), aim_day)
        latest_react_day_app, latest_release_day_app = find_latest_react_and_release_day(aim_day_adjust, update_days_dic[app], app_lag_dic[app])
        if latest_react_day_app < 0:
            continue

        install_app = get_installs(category, app, latest_release_day_app)
        if install_aim != install_app:
            continue

        app_ratings = app_df.loc[(app_df.time >= latest_react_day_app) & (app_df.time <= aim_day_adjust)].rating.tolist()
        # am_app_ratings = amplify_ratings(app_ratings)
        app_median = calculate_rating_statistic(app_ratings)
        # print(app)
        # print('{},{}'.format(latest_react_day_app, aim_day_adjust))
        # print(len(am_app_ratings))
        app_slope, app_increment = calculate_line(range(latest_react_day_app, aim_day_adjust + 1), app_ratings)

        if wholely_above(aim_slope, aim_increment, app_slope, app_increment,
                         latest_react_day_aim, latest_react_day_app, aim_day_adjust):
            better_apps.append([category, app])
            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(range(latest_react_day_aim, aim_day + 1), aim_ratings, color='#fd4659', alpha=0.6)
            ax.plot([latest_react_day_aim, aim_day + 1],
                    [latest_react_day_aim * aim_slope + aim_increment,
                     (aim_day + 1) * aim_slope + aim_increment], color='#fd4659')
            plt.scatter([aim_day + 1], [aim_median], marker='<', color='#fd4659')

            ax.scatter(range(latest_react_day_app, aim_day + 1), app_ratings, color='#87CEFA', alpha=0.6)
            ax.plot([latest_react_day_app, aim_day + 1],
                    [latest_react_day_app * app_slope + app_increment,
                     (aim_day + 1) * app_slope + app_increment], color='#87CEFA')
            plt.scatter([aim_day + 1], [app_median], marker='<', color='#87CEFA')

            save_path = os.path.join(SUPPORT_DIR, str(aim_day), 'Above')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, '%s.png' % app), dpi=200, quality=95)
            # plt.savefig(os.path.join(save_path, '%s.svg' % app), format='svg')
            plt.close()
            '''
        elif not wholely_above(app_slope, app_increment, aim_slope, aim_increment,
                               latest_react_day_app, latest_react_day_aim, aim_day_adjust) and\
            app_slope > aim_slope and app_median > aim_median:
            better_apps.append([category, app])
            '''
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax.scatter(range(latest_react_day_aim, aim_day+1), aim_ratings, color='#fd4659', alpha=0.6)
            ax.plot([latest_react_day_aim, aim_day + 1],
                    [latest_react_day_aim * aim_slope + aim_increment,
                     (aim_day + 1) * aim_slope + aim_increment], color='#fd4659')
            plt.scatter([aim_day + 1], [aim_median], marker='<', color='#fd4659')

            ax.scatter(range(latest_react_day_app, aim_day + 1), app_ratings, color='#87CEFA', alpha=0.6)
            ax.plot([latest_react_day_app, aim_day + 1],
                    [latest_react_day_app * app_slope + app_increment,
                     (aim_day + 1) * app_slope + app_increment], color='#87CEFA')
            plt.scatter([aim_day + 1], [app_median], marker='<', color='#87CEFA')

            save_path = os.path.join(SUPPORT_DIR, str(aim_day), 'Cross')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            plt.savefig(os.path.join(save_path, '%s.png' % app), dpi=200, quality=95)
            # plt.savefig(os.path.join(save_path, '%s.svg' % app), format='svg')
            plt.close()
            '''
    # print(better_apps)
    return better_apps


def filter_app_graph(app_category_dic, aim_app, aim_day, update_days_dic, app_lag_dic):
    better_apps = []
    aim_app_update_days = update_days_dic[aim_app]
    index = 0
    while index < len(aim_app_update_days) and aim_app_update_days[index] < aim_day:
        index += 1

    if index == 0:
        return []

    react_day = aim_app_update_days[index] + app_lag_dic[aim_app]
    while aim_day - react_day < MIN_PERIOD:
        index -= 1
        react_day = aim_app_update_days[index] + app_lag_dic[aim_app]

    aim_df = pandas.read_excel(os.path.join(RATING_DIR, app_category_dic[aim_app], '%s.xlsx' % aim_app))
    aim_ratings = aim_df.loc[(aim_df.time >= react_day) & (aim_df.time <= aim_day)].rating.tolist()
    # am_aim_ratings = amplify_ratings(aim_ratings)
    aim_median = float(calculate_rating_statistic(aim_ratings))
    aim_slope, aim_increment = calculate_line(range(react_day, aim_day + 1), aim_ratings)

    lines = []
    labels = []
    category = 'Social'
    app = 'com_pinterest'
    update_days = update_days_dic[app]
    index = len(update_days) - 1
    while index >= 0 and update_days[index] + app_lag_dic[app] + MIN_PERIOD > aim_day:
        index -= 1
    latest_react_day = update_days[index] + app_lag_dic[app]

    app_df = pandas.read_excel(os.path.join(RATING_DIR, category, '%s.xlsx' % app))
    app_ratings = app_df.loc[(app_df.time >= latest_react_day) & (app_df.time <= aim_day)].rating.tolist()
    # am_app_ratings = amplify_ratings(app_ratings)
    app_median = float(calculate_rating_statistic(app_ratings))
    app_slope, app_increment = calculate_line(range(latest_react_day, aim_day + 1), app_ratings)

    mpl.rcParams['font.sans-serif'] = ['Arial']
    ax1 = plt.subplot2grid((1, 9), (0, 0), colspan=3)
    # ax1.set_xlabel('时间线：距离2016.03.01的天数', fontsize=8, fontproperties="SimHei")
    # ax1.set_ylabel('放大的App评分', fontsize=8, fontproperties="SimHei")
    ax1.set_xlabel('Timeline:the t-th Day from 2016.3.1', fontsize=8)
    ax1.set_ylabel('Rating', fontsize=8)
    # fontproperties = "SimHei"
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # plt.ylim(4, 5)

    ax1.scatter(range(react_day, aim_day+1), aim_ratings, edgecolors='none', color='#fd4659', alpha=0.6)
    line = ax1.plot([react_day, aim_day + 1],
                    [react_day * aim_slope + aim_increment,
                     (aim_day + 1) * aim_slope + aim_increment], color='#fd4659')
    lines.extend(line)
    labels.append('Instagram ({:.2f})'.format(aim_median))
    # plt.scatter([aim_day + 1], [aim_median], marker='<', color='#fd4659')

    ax1.scatter(range(latest_react_day, aim_day + 1), app_ratings, edgecolors='none', color='#6ecb34', alpha=0.6)
    line = ax1.plot([latest_react_day, aim_day + 1],
                    [latest_react_day * app_slope + app_increment,
                     (aim_day + 1) * app_slope + app_increment], color='#6ecb34')
    lines.extend(line)
    labels.append('Pinterest ({:.2f})'.format(app_median))
    # plt.scatter([aim_day + 1], [app_median], marker='<', color='#6ecb34')

    category = 'Games'
    app = 'com_aceviral_smashycity'
    update_days = update_days_dic[app]
    index = len(update_days) - 1
    while index >= 0 and update_days[index] + app_lag_dic[app] + MIN_PERIOD > aim_day:
        index -= 1
    latest_react_day = update_days[index] + app_lag_dic[app]

    app_df = pandas.read_excel(os.path.join(RATING_DIR, category, '%s.xlsx' % app))
    app_ratings = app_df.loc[(app_df.time >= latest_react_day) & (app_df.time <= aim_day)].rating.tolist()
    # am_app_ratings = amplify_ratings(app_ratings)
    app_median = float(calculate_rating_statistic(app_ratings))
    app_slope, app_increment = calculate_line(range(latest_react_day, aim_day + 1), app_ratings)

    ax2 = plt.subplot2grid((1, 9), (0, 4), colspan=3)
    # ax2.set_xlabel('时间线：距离2016.03.01的天数', fontsize=8, fontproperties="SimHei")
    ax2.set_xlabel('Timeline:the t-th Day from 2016.3.1', fontsize=8)
    plt.xticks(fontsize=8)
    plt.yticks(fontsize=8)
    # plt.yticks([])
    # plt.ylim(4, 5)

    ax2.scatter(range(react_day, aim_day + 1), aim_ratings, edgecolors='none', color='#fd4659', alpha=0.6)
    ax2.plot([react_day, aim_day + 1],
             [react_day * aim_slope + aim_increment,
              (aim_day + 1) * aim_slope + aim_increment], color='#fd4659')
    # plt.scatter([aim_day + 1], [aim_median], marker='<', color='#fd4659')

    ax2.scatter(range(latest_react_day, aim_day + 1), app_ratings, edgecolors='none', color='#87CEFA', alpha=0.6)
    line = ax2.plot([latest_react_day, aim_day + 1],
                    [latest_react_day * app_slope + app_increment,
                     (aim_day + 1) * app_slope + app_increment], color='#87CEFA')
    lines.extend(line)
    labels.append('Smashy City ({:.2f})'.format(app_median))
    # plt.scatter([aim_day + 1], [app_median], marker='<', color='#87CEFA')

    ax3 = plt.subplot2grid((1, 9), (0, 7), colspan=2)
    ax3.axis('off')
    ax3.legend(lines, labels, loc='upper left', fontsize='x-small')

    # plt.savefig(os.path.join(save_path, '%s.png' % app), dpi=200, quality=95)
    plt.savefig(os.path.join(SUPPORT_DIR, 'High-rating_Apps.svg'), format='svg')
    # plt.close()

    return better_apps


def get_higher_rating_app(aim_app, aim_day):
    app_category_dic = get_all_app()
    all_apps = list(app_category_dic.keys())

    update_days_dic = {}
    for app in all_apps:
        days = get_update_days(app)
        days = remove_successive_day(days)
        update_days_dic[app] = days

    app_lag_dic = get_apps_lag(all_apps)
    all_apps.remove(aim_app)

    return filter_app(app_category_dic, aim_app, aim_day, update_days_dic, app_lag_dic)


if __name__ == '__main__':
    category = r'Social'
    aim_app = r'com_instagram_android'

    app_category_dic = get_all_app()
    all_apps = list(app_category_dic.keys())

    update_days_dic = {}
    for app in all_apps:
        days = get_update_days(app)
        days = remove_successive_day(days)
        update_days_dic[app] = days

    app_lag_dic = get_apps_lag(all_apps)
    all_apps.remove(aim_app)

    aim_update_days = remove_successive_day(get_update_days(aim_app))
    # filter_app(app_category_dic, aim_app, 100, update_days_dic, app_lag_dic)
    filter_app_graph(app_category_dic, aim_app, 100, update_days_dic, app_lag_dic)




