# -*- coding:utf-8 -*-
'''
Created on 2018.11.16

@author: MollySong
'''

import os
import re
import numpy
from pylab import mpl
from scipy.stats import cumfreq
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.ticker import MultipleLocator

TIMEPAT = re.compile('(?P<day>\d+).txt', re.I)


def sort_func(file_name):
    search_result = TIMEPAT.search(file_name)
    return int(search_result.group('day'))


def get_update_day(pat, file_name):
    search_result = pat.search(file_name)
    return int(search_result.group('day'))


def record_update_slot(apps_dir, record_dir):
    all_slots = []
    f1 = open(os.path.join(record_dir, 'DetailedSlots.txt'), 'w')
    for app_name in os.listdir(apps_dir):
        print(app_name)
        app_dir = os.path.join(apps_dir, app_name)
        _app_name = app_name.replace('.', '_')
        sp_time_pat = re.compile(r'%s(?P<day>\d+).txt' % _app_name, re.I)
        update_days = []
        for file_name in sorted(os.listdir(app_dir), key=sort_func):
            updat_day = get_update_day(sp_time_pat, file_name)
            update_days.append(updat_day)
        update_days.sort()
        update_slots = [update_days[i] - update_days[i - 1] for i in range(1, len(update_days) - 1)]
        f1.write('%s:%s\n' % (app_name, ' '.join(map(str, update_slots))))
        all_slots.extend(update_slots)
    f1.close()
    f2 = open(os.path.join(record_dir, 'Slots.txt'), 'w')
    f2.write(' '.join(map(str, all_slots)))
    f2.close()


def get_slot_num(line):
    slot, num = line.strip('\n').split(':')
    return int(slot), int(num)


def draw_cdf(record_dir):
    f = open(os.path.join(record_dir, 'Slots.txt'), 'r')
    line = f.readline()
    f.close()
    slots = list(map(int, line.strip('\n').split(' ')))

    mpl.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure()

    ax = fig.add_subplot(111)
    xmajorLocator = MultipleLocator(25)
    xminorLocator = MultipleLocator(5)
    ax.xaxis.set_major_locator(xmajorLocator)
    ax.xaxis.set_minor_locator(xminorLocator)
    # plt.grid()
    counts, bin_edges = numpy.histogram(slots, bins=250, normed=True)
    cdf = numpy.cumsum(counts)
    plt.plot(bin_edges[1:], cdf, color='black')

    ax.set_xlabel('Release interval')
    ax.set_ylabel('CDF')
    '''
    ax.set_xlabel('更新间隔', fontproperties="SimHei")
    ax.set_ylabel('App更新间隔累积分布', fontproperties="SimHei")
    '''
    # plt.xticks([i for i in range(1, max(slots), 10)], fontsize=7)

    # plt.show()
    plt.savefig(os.path.join(record_dir, 'CDF_ch.svg'), format='svg')
    plt.close()


def draw_app_slot_distribution(record_dir):
    f = open(os.path.join(record_dir, 'DetailedSlots.txt'), 'r')
    lines = f.readlines()
    f.close()

    all_app = []
    for line in lines:  # one app
        slot_cats = [0, 0, 0]
        app_name, app_slot = line.split(':')
        for slot in map(int, app_slot.split()):
            if slot <= 14:
                slot_cats[0] += 1
            elif slot <= 35:
                slot_cats[1] += 1
            else:
                slot_cats[2] += 1
        all_app.append(slot_cats)
    all_app.sort(key=lambda x: sum(x), reverse=True)

    x_position = 0
    bar_width = 0.5
    # hei_ti = font_manager.FontProperties(fname='C:\Windows\Fonts\simhei.ttf')

    mpl.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_xlabel('App')
    # ax.set_xlabel('移动App', fontproperties="SimHei")
    ax.set_xticks([])
    ax.set_ylabel('Release interval stage distribution')
    # ax.set_ylabel('App更新间隔分布', fontproperties="SimHei")
    '''
    ax.bar(x_position, all_app[0][0], width=bar_width, facecolor='#fd4659',
           label='“半月”类', lw=0)
    ax.bar(x_position, all_app[0][1], width=bar_width, facecolor='#ff964f',
           label='“一月”类', lw=0, bottom=all_app[0][0])
    ax.bar(x_position, all_app[0][2], width=bar_width, facecolor='#6ecb34',
           label='“多月”类', lw=0, bottom=all_app[0][0] + all_app[0][1])
    '''
    ax.bar(x_position, all_app[0][0], width=bar_width, facecolor='#fd4659',
           label='Release interval one', lw=0)
    ax.bar(x_position, all_app[0][1], width=bar_width, facecolor='#ff964f',
           label='Release interval two', lw=0, bottom=all_app[0][0])
    ax.bar(x_position, all_app[0][2], width=bar_width, facecolor='#6ecb34',
           label='Release interval three', lw=0, bottom=all_app[0][0] + all_app[0][1])

    x_position += bar_width
    for slot_cats in all_app[1:]:
        ax.bar(x_position, slot_cats[0], width=bar_width, facecolor='#fd4659', lw=0)
        ax.bar(x_position, slot_cats[1], width=bar_width, facecolor='#ff964f', lw=0, bottom=slot_cats[0])
        ax.bar(x_position, slot_cats[2], width=bar_width, facecolor='#6ecb34', lw=0, bottom=slot_cats[0]+slot_cats[1])
        x_position += bar_width

    ax.legend(loc='upper right')
    # ax.legend(loc='upper right', prop=hei_ti)
    # plt.savefig(os.path.join(record_dir, 'Slot distribution.png'), dpi=200, quality=95)
    plt.savefig(os.path.join(record_dir, 'Slot distribution_ch.svg'), format='svg')
    plt.close()


if __name__ == '__main__':
    # apps_dir = 'D:\programming\workspacePycharm\masterProject\preprocess_whats_new\WhatsNew'
    record_dir = 'D:\programming\workspacePycharm\masterProject\get_supporting_data\Data'

    # record_update_slot(apps_dir, record_dir)
    # draw_cdf(record_dir)
    draw_app_slot_distribution(record_dir)
