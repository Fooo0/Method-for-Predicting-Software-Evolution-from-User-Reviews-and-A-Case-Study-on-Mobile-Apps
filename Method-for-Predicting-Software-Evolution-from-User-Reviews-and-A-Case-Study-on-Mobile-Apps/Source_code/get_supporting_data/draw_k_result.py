# encoding: utf-8
'''
Created on 2019.02.20

@author: Molly Song
'''


import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator


def process_line(line):
    splited_line = line.split(' ')
    return [int(splited_line[0]), float(splited_line[1]), float(splited_line[2])]


FILE_DIR = 'D:\programming\workspacePycharm\masterProject\Data\ChooseK'
K_DIR = r'D:\programming\workspacePycharm\masterProject\Data\ChooseK'


def draw_files(file_list, f_name):
    all_data = []
    ks = []
    sses = []
    chis = []
    for file in file_list:
        one_file = open(os.path.join(FILE_DIR, file), 'r')
        lines = one_file.readlines()
        datas = list(map(process_line, lines))
        all_data.extend(datas)
    all_data.sort()

    for d in all_data:
        ks.append(d[0])
        sses.append(d[1])
        chis.append(d[2])

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xmajorLocator = MultipleLocator(100)
    xminorLocator = MultipleLocator(10)
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_minor_locator(xminorLocator)

    ax1.set_xlabel('K')
    ax1.set_ylabel('SSE')
    # ax1.set_xticks(k_candidate)
    ax1.plot(ks, sses, color='#87CEFA')
    plt.xticks(fontsize=6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Calinski-Harabasz Index')
    ax2.plot(ks, chis, color='#f36198')

    plt.savefig(os.path.join(K_DIR, '%s.svg' % f_name), format='svg')
    print('SAVE')


if __name__ == '__main__':
    file_list = ['100_6000_100_500.txt', '810_1100_10.txt']
    f_name = 'BB'
    draw_files(file_list, f_name)
