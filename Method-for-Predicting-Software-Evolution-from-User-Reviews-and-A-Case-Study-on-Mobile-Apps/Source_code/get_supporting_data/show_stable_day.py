# -*- coding:utf-8 -*-
'''
Created on 2019.04.02

@author: MollySong
'''

import os
import pandas
import seaborn
from pylab import mpl
from collections import Counter
import matplotlib.pyplot as plt


SERIAL_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Serial_predict'
CATEGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'


def get_max_entropies():
    max_entropies = []
    for category in ['Entertainment', 'Games']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(CATEGORY_PATH):
        category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)
            df_path = os.path.join(SERIAL_DIR, category, app_name, 'UC_entropy_b.xlsx')
            if not os.path.exists(df_path):
                continue
            df = pandas.read_excel(df_path)
            one_max = df['Average_entropy'].max()
            max_entropies.append(one_max)
    return max_entropies


if __name__ == '__main__':
    data_path = os.path.join(SERIAL_DIR, 'UC_min_day.xlsx')
    df = pandas.read_excel(data_path)
    stable_days = df['Min_day'].tolist()
    max_entropies = get_max_entropies()

    mpl.rcParams['font.sans-serif'] = ['Arial']

    ax1= plt.subplot2grid((1, 9), (0, 0), colspan=3)
    seaborn.swarmplot(data=max_entropies, ax=ax1, color='lightgray')
    plt.xlabel('(a)')
    plt.xticks([])
    plt.ylabel('Max entropy')

    ax2 = plt.subplot2grid((1, 9), (0, 4), colspan=3)
    seaborn.swarmplot(data=stable_days, ax=ax2, color='lightgray')
    # ax = seaborn.boxplot(data=stable_days, width=0.3, fliersize=2,
    #                      boxprops={'color': 'black', 'facecolor': 'lightgray'})
    # plt.bar(days, counts)
    # seaborn.violinplot(stable_days, orient='v', inner='point', color='lightgray')
    plt.xlabel('(b)')
    plt.xticks([])
    plt.ylabel('The number of days since the last release')

    # plt.savefig(os.path.join(SERIAL_DIR, 'UC_entropy_with_day.svg'), format='svg')
    # plt.close()
    plt.show()
