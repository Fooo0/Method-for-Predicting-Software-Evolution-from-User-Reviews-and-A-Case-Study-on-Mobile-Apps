# -*- coding:utf-8 -*-
'''
Created on 2018.11.29

@author: MollySong
'''


import numpy
from pylab import mpl
import matplotlib.pyplot as plt
from matplotlib import font_manager

if __name__ == '__main__':
    m_rake_precision = [0.776, 0.777, 0.738, 0.6824, 0.903, 0.833]  # 78.5%
    m_rake_recall = [0.784, 0.725, 0.796, 0.761, 0.644, 0.747]  # 74.3

    rake_precision = [0.63, 0.612, 0.6232, 0.55, 0.71, 0.732]  # 64.3%
    rake_recall = [0.773, 0.72, 0.782, 0.776, 0.656, 0.683]  # 73.2%

    mpl.rcParams['font.sans-serif'] = ['Arial']
    # hei_ti = font_manager.FontProperties(fname='C:\Windows\Fonts\simhei.ttf')
    fig = plt.figure(figsize=(10, 3.7))
    fig.subplots_adjust(right=0.97, bottom=0.2, top=0.71)

    ax = fig.add_subplot(111)
    x_tickets = ['Ebay', 'Skype', 'Pandora', 'Waze', 'Weather', 'Instagram Layout']

    total_width, n = 0.6, 4
    width = total_width / n
    x = numpy.arange(len(x_tickets))
    x = x - (total_width - width) / 2

    plt.ylim(0.5, 1.0)

    plt.bar(x, m_rake_precision, width=width, label='Precision of mRAKE', color='#fd4659')
    plt.bar(x + width, m_rake_recall, width=width, label='Recall of mRAKE', color='#fe828c')
    plt.bar(x + 2*width, rake_precision, width=width, label='Precision of RAKE', color='#1E90FF')
    plt.bar(x + 3*width, rake_recall, width=width, label='Recall of RAKE', color='#87CEFA')
    '''
    plt.bar(x, m_rake_precision, width=width, label='改进后准确率', color='#fd4659')
    plt.bar(x + width, m_rake_recall, width=width, label='改进后召回率', color='#fe828c')
    plt.bar(x + 2 * width, rake_precision, width=width, label='改进前准确率', color='#1E90FF')
    plt.bar(x + 3 * width, rake_recall, width=width, label='改进前召回率', color='#87CEFA')
    '''
    plt.xticks(x + 2*width, x_tickets)

    plt.title('Comparison between mRAKE and RAKE')
    ax.set_xlabel('App')
    ax.set_ylabel('Precision and recall')
    ax.legend(ncol=4)
    '''
    plt.title('RAKE算法改进前后特征提取实验结果对比', fontproperties="SimHei")
    ax.set_xlabel('移动App', fontproperties="SimHei")
    ax.set_ylabel('准确率 和 召回率', fontproperties="SimHei")
    ax.legend(ncol=4, prop=hei_ti)
    '''
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    plt.savefig(r'D:\programming\workspacePycharm\masterProject\get_supporting_data\Data\Feature_extraction_efficiency.svg',
                format='svg')
    plt.close()
