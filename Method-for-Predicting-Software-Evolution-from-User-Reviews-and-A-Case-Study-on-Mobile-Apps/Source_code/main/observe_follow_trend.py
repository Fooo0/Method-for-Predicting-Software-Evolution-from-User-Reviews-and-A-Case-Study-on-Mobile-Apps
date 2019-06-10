# -*- coding:utf-8 -*-
'''
Created on 2018.12.27

@author: Molly Song
'''


import numpy
import matplotlib.pyplot as plt


def calculate_min_and_avg_update_slot(times):
    slots = [times[i]-times[i-1] for i in range(1, len(times))]
    return numpy.min(slots), numpy.mean(slots)


if __name__ == '__main__':
    x_iheart = [31, 54, 61, 66, 131, 184, 214, 253, 283, 311, 323, 325, 343]
    min_iheart, avg_iheart = calculate_min_and_avg_update_slot(x_iheart)

    x_pandora = [229, 235, 257, 381]
    min_pandora, avg_pandora = calculate_min_and_avg_update_slot(x_pandora)

    fig = plt.figure()
    ax = fig.add_subplot(111)

    plt.vlines([x_iheart[0]], 0, 1, color='#fd4659', label='iheart radio: %f/%f' % (min_iheart, avg_iheart))
    for x in x_iheart[1:]:
        plt.vlines(x, 0, 1, color='#fd4659')

    plt.vlines([x_pandora[0]], 0, 1, color='#87CEFA', label='pandora radio: %f/%f' % (min_pandora, avg_pandora))
    for x in x_pandora[1:]:
        plt.vlines(x, 0, 1, color='#87CEFA')

    ax.legend(loc='upper right')
    plt.xticks(x_iheart + x_pandora)
    plt.ylim(0, 2)
    plt.yticks([0, 1, 2])
    plt.show()
