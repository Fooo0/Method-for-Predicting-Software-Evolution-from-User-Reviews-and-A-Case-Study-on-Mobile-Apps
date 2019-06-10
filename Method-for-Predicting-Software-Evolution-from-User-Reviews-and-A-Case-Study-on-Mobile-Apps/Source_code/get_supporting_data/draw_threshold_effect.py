# -*- coding:utf-8 -*-
'''
Created on 2019.03.17

@author: MollySong
'''


import os
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


THRESHOLD_EFFECT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result\Threshold_effect'


def get_data(file_path):
    f = open(file_path, 'r')
    lines = f.readlines()
    f.close()
    ts = []
    ss = []
    for l in lines:
        t, s = list(map(float, l.strip('\n').split(' ')))
        ts.append(round(abs(t), 2))
        ss.append(s)
    return ts, ss


def draw_single_varyging_accuracy(app_name, file, title, y_label, color, svg_name):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    fig = plt.figure()
    ax = fig.add_subplot(111)

    x, y = get_data(os.path.join(THRESHOLD_EFFECT_DIR, app_name, file))
    plt.xticks(x)
    ax.plot(x, y, '-', color=color)
    plt.title(title)
    plt.xlabel("Kendall's tau coefficient threshold")
    plt.ylabel(y_label)
    # plt.show()
    plt.savefig(os.path.join(THRESHOLD_EFFECT_DIR, app_name, svg_name), format='svg')
    plt.close()


def draw_multiple_varyging_accuracy(app_name, file_list, labels, title, y_label, colors, svg_name):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    # x_major_locator = MultipleLocator(0.25)
    # x_minor_locator = MultipleLocator(0.05)

    fig, axes = plt.subplots(len(file_list), 1, sharex='col')
    for i in range(len(file_list)):
        x, y = get_data(os.path.join(THRESHOLD_EFFECT_DIR, app_name, file_list[i]))
        # axes[i].xaxis.set_major_locator(x_major_locator)
        # axes[i].xaxis.set_minor_locator(x_minor_locator)
        axes[i].plot(x, y, '-', label=labels[i], color=colors[i])
        axes[i].legend(loc=3)
    # plt.show()
    axes[0].set_title(title)
    axes[2].set_xlabel("Kendall's tau coefficient threshold")
    axes[1].set_ylabel(y_label)
    plt.savefig(os.path.join(THRESHOLD_EFFECT_DIR, app_name, svg_name), format='svg')
    plt.close()


if __name__ == '__main__':

    app_name = r'com_clearchannel_iheartradio_controller'
    feature = 'power optimising'
    title = '"{}" of iHeartRadio'.format(feature)
    y_label = "Average prediction accuracy"
    draw_single_varyging_accuracy(app_name, 'UC_power_optimising.txt', title, y_label, 'black',
                                  'UC_power_optimising.svg')
    
    app_name = r'com_smule_singandroid'
    y_label = "Average prediction accuracy"
    draw_single_varyging_accuracy(app_name, 'UT.txt', app_name, y_label, 'black', 'UT.svg')

    app_name = r'com_smule_singandroid'
    postfixes = ['freq', 'pos', 'neg']
    file_list = []
    for postfix in postfixes:
        file_list.append('US_{}.txt'.format(postfix))
    labels = ['Intensity', 'Pos. sentiment score', 'Neg. sentiment score']
    y_label = "Average prediction accuracy"
    colors = ['#87CEFA', '#fd4659', '#6ecb34']
    draw_multiple_varyging_accuracy(app_name, file_list, labels, app_name, y_label, colors, 'US.svg')
