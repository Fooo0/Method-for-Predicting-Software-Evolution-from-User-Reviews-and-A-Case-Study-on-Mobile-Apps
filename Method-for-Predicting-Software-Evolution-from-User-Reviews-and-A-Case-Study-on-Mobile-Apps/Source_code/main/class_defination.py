# encoding: utf-8
'''
Created on 2017.10.09

@author: Molly Song
'''


import os
import re
import numpy
import shutil
from pylab import mpl
import matplotlib.pyplot as plt
from collections import defaultdict
from scipy.spatial.distance import euclidean
from matplotlib.ticker import MultipleLocator


MIN_PERIOD = 2


class FeatureCase(object):
    def __init__(self, app_name, feature, f_vecotr, freq, senti_score, time, isRev):
        self.app_name = app_name
        self.feature = feature
        self.f_vecotr = f_vecotr
        self.freq = freq
        self.senti_score = senti_score
        self.time = time
        self.isRev = isRev

    def get_app_name(self):
        return self.app_name

    def set_time(self, time):
        self.time = time

    def get_time(self):
        return self.time

    def get_feature(self):
        return self.feature

    def get_vector(self):
        return self.f_vecotr

    def get_freq(self):
        return self.freq

    def get_senti_score(self):
        return self.senti_score

    def is_review(self):
        return self.isRev


class ClusterPlotter(object):
    def __init__(self, category_dic, one_big_cluster, centroid, times_rev_dic, times_whatsnew_dic,
                 successive_daysWh, successive_dict):
        self.illegal_mark = re.compile(r"[\\/:*?\"<>\|]")
        self.category_dic = category_dic
        self.one_cluster = one_big_cluster
        self.centroid = centroid
        self.times_rev_dic = times_rev_dic
        self.times_whatsnew_dic = times_whatsnew_dic
        self.features_r = []
        self.features_w = []
        self.successive_daysWh = successive_daysWh
        self.successive_dict = successive_dict
        self.label = None
        self.timed_cluster_r = defaultdict(lambda: defaultdict(lambda: [0, [], []]))  # app: time: [freq, [pos], [neg]]
        self.timed_cluster_w = defaultdict(lambda: defaultdict(lambda: 0))
        self.apps = []

    def detail_cluster(self, path):
        one_file = open(path, 'a')
        one_file.write("================ ONE CLUSTER START ================>\n")
        min_distance = None  # find the representative of the cluster
        candi_label = None
        for item in self.one_cluster:
            if item.is_review():
                self.features_r.append(item)
            else:
                one_file.write("W\t")
                self.features_w.append(item)
            one_freq = item.get_freq()
            feat_label = ' '.join(item.get_feature())
            one_file.write("{}:{}\n".format(feat_label.encode('utf-8'), one_freq))
            feat_vector = item.get_vector()
            distance = euclidean(self.centroid, feat_vector)  # calculate the euclidean distance between item and centroid
            if not self.illegal_mark.search(feat_label) and (not min_distance or distance < min_distance):
                min_distance = distance
                candi_label = feat_label
            self.apps.append(item.get_app_name())
        self.label = candi_label
        one_file.write("<================ NAME: {} ================\n".format(self.label))
        one_file.close()
        self.apps = list(set(self.apps))
        return

    def time_cluster(self):
        for item in self.features_r:
            time = item.get_time()
            freq = item.get_freq()
            app_name = item.get_app_name()
            scores = [float(i) for i in item.get_senti_score().split()]
            self.timed_cluster_r[app_name][time][0] += freq
            for score in scores:
                if score >= 0:
                    self.timed_cluster_r[app_name][time][1].append(score)  # sum of positive score
                elif score < 0:
                    self.timed_cluster_r[app_name][time][2].append(score)  # sum of negative score

        for item in self.features_w:
            time = item.get_time()
            app_name = item.get_app_name()
            if time in self.successive_dict:
                time = self.successive_dict[time]
                item.set_time(time)
            # freq = item.get_freq()
            if time not in self.successive_daysWh:
                # self.timedWhsNewCluster[time] += freq
                self.timed_cluster_w[app_name][time] = 1
        return

    def plot_cluster(self, save_dir):
        for app in self.apps:
            values_rev = []
            feat_updated_time = []
            feat_not_updated_time = []
            values_pos_senti = []
            values_neg_senti = []
            for time in self.times_rev_dic[app]:
                values_rev.append(self.timed_cluster_r[app][time][0])

                p_lis = self.timed_cluster_r[app][time][1]  # positive sentiment score
                if p_lis:
                    values_pos_senti.append(numpy.mean(p_lis))
                else:
                    values_pos_senti.append(0)

                n_lis = self.timed_cluster_r[app][time][2]  # negative sentiment score
                if n_lis:
                    values_neg_senti.append(numpy.mean(n_lis))
                else:
                    values_neg_senti.append(0)
            for time in self.times_whatsnew_dic[app]:
                if self.timed_cluster_w[app][time]:
                    feat_updated_time.append(time)
                else:
                    feat_not_updated_time.append(time)

            mpl.rcParams['font.sans-serif'] = ['Arial']

            fig = plt.figure()
            fig.subplots_adjust(left=0.1, bottom=0.01, right=0.97, top=0.95)

            ax1 = fig.add_subplot(211)
            x_major_locator = MultipleLocator(10)
            x_minor_locator = MultipleLocator(1)
            ax1.xaxis.set_major_locator(x_major_locator)
            ax1.xaxis.set_minor_locator(x_minor_locator)
            ax1.spines['top'].set_color('none')
            xticket = [t for t in range(min(self.times_rev_dic[app]),
                                        max(self.times_rev_dic[app]) + 1) if t % 10 == 0]
            plt.xticks(xticket, rotation=45, fontsize=6)
            plt.yticks(fontsize=7)
            plt.title(self.label, fontsize=7)
            plt.ylim(0, max(values_rev))
            ax1.set_xlabel('Timeline:the t-th Day from 2016.3.1', fontsize=6)
            ax1.set_ylabel('Intensity', fontsize=6)

            ax1.plot(self.times_rev_dic[app], values_rev, '-', label='Intensity',
                     linewidth=0.7, color='#87CEFA')
            ax1.scatter(feat_updated_time, [0 for i in range(len(feat_updated_time))], s=20,
                        label='Updated in this version', marker='+', c='black')
            ax1.scatter(feat_not_updated_time, [0 for j in range(len(feat_not_updated_time))],
                        s=20, label='Not updated in this version', marker='x', c='black')
            ax1.legend(loc='upper right', prop={'size': 6}, framealpha=0.3)

            ax2 = fig.add_subplot(212)
            ax2.xaxis.set_major_locator(x_major_locator)
            ax2.xaxis.set_minor_locator(x_minor_locator)
            ax2.spines['top'].set_color('none')
            ax2.spines['bottom'].set_position(('data', 0))
            plt.xticks(xticket, rotation=45, fontsize=6)
            plt.yticks(fontsize=7)
            plt.ylim(min(values_neg_senti), max(values_pos_senti))
            ax2.set_ylabel('Sentiment score', fontsize=6)

            ax2.plot(self.times_rev_dic[app], values_pos_senti, '-', label='Avg. pos. sentiment score',
                     linewidth=0.7, color='#fd4659')
            ax2.plot(self.times_rev_dic[app], values_neg_senti, '-', label='Avg. neg. sentiment score',
                     linewidth=0.7, color='#6ecb34')
            ax2.legend(loc='upper right', prop={'size': 6}, framealpha=0.3)

            # plt.savefig(os.path.join(save_path, self.label + '.png'), dpi=200)
            plot_dir = os.path.join(save_dir, self.category_dic[app], app)
            if not os.path.exists(plot_dir):
                os.makedirs(plot_dir)
            plt.savefig(os.path.join(plot_dir, '%s.svg' % self.label), format='svg')
            plt.close()
        return

    def write_to_file(self, dir1, dir2):
        for app in self.apps:
            dir_app = os.path.join(dir1, self.category_dic[app], app, self.label)
            if not os.path.exists(dir_app):
                os.makedirs(dir_app)
            file_rev = open(os.path.join(dir_app, 'Review.txt'), 'a')
            for time in self.times_rev_dic[app]:
                p_lis = self.timed_cluster_r[app][time][1]  # positive sentiment score
                if p_lis:
                    p_mean = numpy.mean(p_lis)
                else:
                    p_mean = 0
                n_lis = self.timed_cluster_r[app][time][2]  # negative sentiment score
                if n_lis:
                    n_mean = numpy.mean(n_lis)
                else:
                    n_mean = 0
                file_rev.write("%d:::%f:::%f:::%f\n" % (time, self.timed_cluster_r[app][time][0], p_mean, n_mean))

            file_whsnew = open(os.path.join(dir_app, 'WhatsNew.txt'), 'a')
            has_zero = False
            has_one = False
            for time in self.times_whatsnew_dic[app]:
                update_flag = self.timed_cluster_w[app][time]
                file_whsnew.write("%d:::%d\n" % (time, self.timed_cluster_w[app][time]))
                if not has_one and update_flag == 1 and (time-self.times_whatsnew_dic[app][0]) >= MIN_PERIOD+1:
                    has_one = True
                if not has_zero and update_flag == 0:
                    has_zero = True

            file_rev.close()
            file_whsnew.close()

            if has_zero and has_one:
                shutil.copytree(dir_app,
                                os.path.join(dir2, self.category_dic[app], app, self.label))
        return
