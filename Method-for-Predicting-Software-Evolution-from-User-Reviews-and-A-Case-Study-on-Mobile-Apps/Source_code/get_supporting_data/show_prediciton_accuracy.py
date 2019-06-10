# -*- coding:utf-8 -*-
'''
Created on 2019.02.28

@author: MollySong
'''


import os
import numpy
import pandas
import seaborn
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def gather_line_data(line, list_1, list_2, list_3):
    figures = list(map(float, line.split(' ')))
    if figures[0] != 0.0 and figures[0] != 1.0:
        list_1.append(figures[0])
    if figures[1] != 0.0 and figures[0] != 1.0:
        list_2.append(figures[1])
    if figures[2] != 0.0 and figures[0] != 1.0:
        list_3.append(figures[2])


CATEGORY_PATH = r'D:\programming\workspacePycharm\masterProject\AppCategory'
PREDICTION_RESULT_DIR = r'D:\programming\workspacePycharm\masterProject\Data\Predict_result'


def get_user_satisfaction_accuracies():
    threshold_freq = []
    threshold_pos = []
    threshold_neg = []

    accuracy_freq = []
    accuracy_pos = []
    accuracy_neg = []
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    #     file_name = '{}.txt'.format(category)
    for file_name in os.listdir(CATEGORY_PATH):
        category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)

            accuracy_path = os.path.join(PREDICTION_RESULT_DIR, category, app_name, 'US.txt')
            if not os.path.exists(accuracy_path):
                continue

            f = open(accuracy_path, 'r')
            lines = f.readlines()
            f.close()
            gather_line_data(lines[0].strip('\n'), threshold_freq, threshold_pos, threshold_neg)
            gather_line_data(lines[1].strip('\n'), accuracy_freq, accuracy_pos, accuracy_neg)
    '''
    df1 = pandas.DataFrame({u'用户评论强度': accuracy_freq})
    df2 = pandas.DataFrame({u'正情感值': accuracy_pos})
    df3 = pandas.DataFrame({u'负情感值': accuracy_neg})
    '''
    df1 = pandas.DataFrame({'Intensity': accuracy_freq})
    df2 = pandas.DataFrame({'Pos. sentiment score': accuracy_pos})
    df3 = pandas.DataFrame({'Neg. sentiment score': accuracy_neg})

    df4 = pandas.DataFrame({'Intensity': threshold_freq})
    df5 = pandas.DataFrame({'Pos. sentiment score': threshold_pos})
    df6 = pandas.DataFrame({'Neg. sentiment score': threshold_neg})
    print(numpy.mean(accuracy_freq))
    print(numpy.mean(accuracy_pos))
    print(numpy.mean(accuracy_neg))
    return [df1, df2, df3], [df4, df5, df6]


def get_accuracy_from_line(line):
    return numpy.mean(list(map(float, line.strip('\n').split(' '))))
    # return float(line.strip('\n'))


def get_update_content_accuracies(df_name, UC_file_name):
    accuracies = []
    for category in ['Books & Reference', 'Business', 'Education',
                     'Social', 'Communication', 'Finance', 'Maps & Navigation',
                     'News & Magazines', 'Travel & Local']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(CATEGORY_PATH):
    #     category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)

            accuracy_path = os.path.join(PREDICTION_RESULT_DIR, category, app_name, UC_file_name)
            if not os.path.exists(accuracy_path):
                continue

            f = open(accuracy_path, 'r')
            lines = f.readlines()
            f.close()

            accuracies.extend(list(map(get_accuracy_from_line, lines[:-1])))
    df = pandas.DataFrame({df_name: accuracies})
    print(numpy.mean(accuracies))
    return df


def get_update_time_accuracies():
    accuracies = []
    thresholds = []
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    #     file_name = '{}.txt'.format(category)
    for file_name in os.listdir(CATEGORY_PATH):
        category = file_name.split('.')[0]
        print(category)
        f = open(os.path.join(CATEGORY_PATH, file_name), 'r')
        app_names = f.readlines()
        f.close()
        for app_name in app_names:
            app_name = app_name.strip('\n')
            print(app_name)

            accuracy_path = os.path.join(PREDICTION_RESULT_DIR, category, app_name, 'UT.txt')
            if not os.path.exists(accuracy_path):
                continue

            f = open(accuracy_path, 'r')
            line = f.readline()
            f.close()

            t, a = list(map(float, line.strip('\n').split(' ')))
            thresholds.append(t)
            accuracies.append(a)

    df1 = pandas.DataFrame({'Update time': accuracies})
    df2 = pandas.DataFrame({'Update time': thresholds})
    print(numpy.mean(accuracies))
    return df1, df2


def draw_odd_boxplots(df_list, x_label, y_label, colors, file_name):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['axes.unicode_minus'] = False

    fig, axes = plt.subplots(1, len(df_list), sharey='row')

    axes[int(len(df_list) / 2)].set_xlabel(x_label, fontsize='large')
    axes[0].set_ylabel(y_label, fontsize='large')
    y_major_locator = MultipleLocator(0.1)
    y_minor_locator = MultipleLocator(0.01)

    for i in range(len(df_list)):
        # ax = fig.add_subplot(1, len(df_list), i)
        axes[i].yaxis.set_major_locator(y_major_locator)
        axes[i].yaxis.set_minor_locator(y_minor_locator)
        seaborn.boxplot(data=df_list[i], width=0.5, ax=axes[i],
                        boxprops={'color': 'black', 'facecolor': colors[i]})

    plt.savefig(os.path.join(PREDICTION_RESULT_DIR, '{}.svg'.format(file_name)), format='svg')
    plt.close()


def draw_single_boxplot(df, x_label, y_label, color, file_name):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    # mpl.rcParams['axes.unicode_minus'] = False
    ax = seaborn.boxplot(data=df, width=0.3, fliersize=2,
                         boxprops={'color': 'black', 'facecolor': color})
    '''
    plt.xlabel(x_label, fontsize='large', fontproperties="SimHei")
    plt.ylabel(y_label, fontsize='large', fontproperties="SimHei")
    '''
    plt.xlabel(x_label, fontsize='large')
    plt.ylabel(y_label, fontsize='large')

    y_major_locator = MultipleLocator(0.1)
    y_minor_locator = MultipleLocator(0.01)
    ax.yaxis.set_major_locator(y_major_locator)
    ax.yaxis.set_minor_locator(y_minor_locator)
    # plt.xticks(['Update content'])

    plt.savefig(os.path.join(PREDICTION_RESULT_DIR, '{}.svg'.format(file_name)), format='svg')
    plt.close()


def draw_even_boxplots(df_list, x_label, y_label, colors, file_name):
    mpl.rcParams['font.sans-serif'] = ['Arial']
    # mpl.rcParams['font.sans-serif'] = ['SimHei']
    # mpl.rcParams['axes.unicode_minus'] = False
    y_major_locator = MultipleLocator(0.1)
    y_minor_locator = MultipleLocator(0.01)

    ax1 = plt.subplot2grid((1, 9), (0, 0), colspan=4)
    ax1.yaxis.set_major_locator(y_major_locator)
    ax1.yaxis.set_minor_locator(y_minor_locator)
    seaborn.boxplot(data=df_list[0], width=0.5, ax=ax1,
                    boxprops={'color': 'black', 'facecolor': colors[0]})
    ax1.set_ylabel(y_label, fontsize='large')
    ax1.set_xlabel(x_label, fontsize='large', horizontalalignment='left')

    ax2 = plt.subplot2grid((1, 9), (0, 4), sharey=ax1)
    ax2.axis('off')

    ax3 = plt.subplot2grid((1, 9), (0, 5), colspan=4, sharey=ax1)
    ax3.yaxis.set_major_locator(y_major_locator)
    ax3.yaxis.set_minor_locator(y_minor_locator)
    seaborn.boxplot(data=df_list[1], width=0.5, ax=ax3,
                    boxprops={'color': 'black', 'facecolor': colors[1]})

    plt.savefig(os.path.join(PREDICTION_RESULT_DIR, '{}.svg'.format(file_name)), format='svg')
    plt.close()


if __name__ == '__main__':
    '''
    df_list1, df_list2 = get_user_satisfaction_accuracies()
    # draw_odd_boxplots(df_list1, '用户满意度', '预测准确率',
    #                       ['#87CEFA', '#fd4659', '#6ecb34'], 'US_Accuracy_ch')
    draw_odd_boxplots(df_list1, 'Prediction target', 'Prediction accuracy',
                          ['#87CEFA', '#fd4659', '#6ecb34'], 'US_Accuracy')
    # draw_user_satisfaction_boxplot(df_list2, 'Prediction target', 'Kendall tau rank correlation coefficient threshold',
    #              ['#87CEFA', '#fd4659', '#6ecb34'], 'US_Threshold')
    '''

    df1 = get_update_content_accuracies('Update content', 'UC_accuracy.txt')
    # df2 = get_update_content_accuracies('User-review based method\n+\nHigh-rating similar App based method', 'UC_accuracy_.txt')
    # draw_even_boxplots([df1, df2], '                Prediction method', 'Prediction accuracy',
    #                    ['#87CEFA', '#fd4659'], 'UC_Accuracy_contrast')
    draw_single_boxplot(df1, 'Prediction target', 'Prediction accuracy', 'lightgray', 'UC_Accuracy')
    # draw_single_boxplot(df1, 'App更新内容', '预测准确率', 'white', 'UC_Accuracy_ch')
    # draw_single_boxplot(df2, 'App更新内容', '预测准确率', 'white', 'UC_Accuracy__ch')

    '''
    df1, df2 = get_update_time_accuracies()
    # draw_single_boxplot(df1, 'App更新时间', '预测准确率', 'white', 'UT_Accuracy_ch')
    draw_single_boxplot(df1, 'Prediction target', 'Prediction accuracy', 'lightgray', 'UT_Accuracy')
    # draw_single_boxplot(df2, 'Prediction target', 'Kendall tau rank correlation coefficient threshold',
    #                     '#87CEFA', 'UT_Threshold')
    '''
