# -*- coding:utf-8 -*-
'''
Created on 2017.10.09

@author: Molly Song
'''


import re
import os
import gc
import gensim
import traceback
import numpy
from numpy import float32
from sklearn import metrics
from sklearn.cluster import MiniBatchKMeans
from main.class_defination import FeatureCase, ClusterPlotter


def remove_successive_days(times_whatsnew):
    out = []
    dic = {}
    lenWh = len(times_whatsnew)
    for i in range(lenWh):
        if i + 1 < lenWh and times_whatsnew[i] + 1 == times_whatsnew[i + 1]:
            out.append(times_whatsnew[i + 1])
            dic[times_whatsnew[i + 1]] = times_whatsnew[i]
    for i in out:
        times_whatsnew.remove(i)
    return out, dic


KEYWORD_DIR = r'D:\programming\workspacePycharm\MasterProject\Data\Keywords'
# KEYWORD_DIR = r'D:\programming\workspacePycharm\masterProject\Test\Data\Keywords'
CLUSTER_DETAIL_PATH = r'D:\programming\workspacePycharm\masterProject\Data\Cluster_detail.txt'
# CLUSTER_DETAIL_PATH = r'D:\programming\workspacePycharm\masterProject\Test\Data\Cluster_detail.txt'
CLUSTER_DATA_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataForPrediction'
# CLUSTER_DATA_DIR = r'D:\programming\workspacePycharm\masterProject\Test\Data\DataForPrediction'
PICKED_CLUSTER_DATA_DIR = r'D:\programming\workspacePycharm\masterProject\Data\DataPredictable'
# PICKED_CLUSTER_DATA_DIR = r'D:\programming\workspacePycharm\masterProject\Test\Data\DataPredictable'
CLUSTER_IMAGE_DIR = r'D:\programming\workspacePycharm\masterProject\Data\ClusterImage'
# CLUSTER_IMAGE_DIR = r'D:\programming\workspacePycharm\masterProject\Test\Data\ClusterImage'
MODEL_PATH = r'D:\programming\workspacePycharm\masterProject\Word2VectorModel\wiki.en.text.vector'
K_DIR = r'D:\programming\workspacePycharm\masterProject\Data\ChooseK'
THRESHOLD = 10e-7
K = 800


def cluster_apps(catg_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    model_vectors = model.wv
    del model
    gc.collect()
    category_dic = {}
    times_rev_dic = {}
    times_whatsnew_dic = {}
    successive_daysWh = {}
    successive_dict = {}
    all_vec = []
    # all_vector_array = numpy.zeros((8160925, 200), dtype=float32)
    # array_row = 0
    all_feature_cases = []
    # for category in ['Entertainment', 'Games']:
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    for category in ['Music & Audio', 'Photography', 'Personalization',
                     'Productivity', 'Tools', 'Weather', 'Lifestyle']:
    # for category in ['Shopping']:
        file_name = '{}.txt'.format(category)
        with open(os.path.join(catg_path, file_name), 'r') as f:
            for an in f:
                an = an.strip('\n')
                print(an)
                category_dic[an] = category
                source_data_dir = [(os.path.join(KEYWORD_DIR, 'Whatsnew', an), False),
                                   (os.path.join(KEYWORD_DIR, 'Review', an), True)]
                last_time_rev = -1
                times_rev = []
                times_whatsnew = []
                for sdd, flag in source_data_dir:
                    for fileName in sorted(os.listdir(sdd), key=lambda x: int(x.split(".")[0])):
                        time_day = int(fileName.split(".")[0])  # get date
                        print(time_day)
                        if flag:
                            while last_time_rev > 0 and (time_day - last_time_rev) != 1:
                                last_time_rev += 1
                                times_rev.append(last_time_rev)
                            times_rev.append(time_day)
                            last_time_rev = time_day
                        else:
                            times_whatsnew.append(time_day)
                        with open(os.path.join(sdd, fileName), 'r',
                                  encoding='utf-8', errors='ignore') as one_file:
                            for line in one_file:  # one feature information
                                line = line.strip('\n')
                                splited_line = line.split(':::')
                                if not splited_line[0]:
                                    continue
                                feature = splited_line[0].split()  # get feature
                                feature_vector = numpy.zeros(shape=(200,), dtype=float32)
                                strange_w = False
                                for w in feature:
                                    w = re.sub(':', '', w)
                                    try:
                                        feature_vector += model_vectors.get_vector(w).astype(float32)
                                        # word_vectors.append(model_vectors.get_vector(w))
                                    except KeyError:
                                        strange_w = True
                                        break
                                if strange_w:
                                    continue
                                all_vec.append(feature_vector)
                                # all_vector_array[array_row, :] = feature_vector
                                # array_row += 1
                                try:
                                    if flag:
                                        freq = float(re.sub(':', '', splited_line[1]))  # feature frequency
                                        # freq_check += freq
                                    else:
                                        freq = 1
                                except ValueError:
                                    print(splited_line)
                                    traceback.print_exc()
                                    exit(0)
                                senti_scores = re.sub(':', '', splited_line[-1].strip('\n'))
                                one_feature_case = FeatureCase(an, feature, feature_vector, freq, senti_scores, time_day, flag)
                                all_feature_cases.append(one_feature_case)  # allUpdPrd structure : [FeatureCase, FeatureCases,...]

                times_rev_dic[an] = sorted(times_rev)
                times_whatsnew_dic[an] = sorted(times_whatsnew)
                successive_daysWh[an], successive_dict[an] = remove_successive_days(times_whatsnew)

    print('Start cluster')
    kmeans = MiniBatchKMeans(n_clusters=K, max_iter=300, batch_size=80000, n_init=10).fit(all_vec)  # use euclidean metric to measure similarity
    result_labels = kmeans.labels_
    chi = metrics.calinski_harabaz_score(all_vec, kmeans.labels_)  # Calinski-Harabasz Index
    # sil = metrics.silhouette_score(all_vec, kmeans.labels_)  # silhouette coefficient
    print("Evaluation:\n\tCalinski-Harabasz Index = %f.\n\tSSE = %f.\n" % (chi, kmeans.inertia_))

    clusters = []
    for i in range(K):  # structure: [[cluster], [clusters], ...]
        clusters.append([])

    for index, fc in zip(list(result_labels), all_feature_cases):
        clusters[index].append(fc)

    print("Start to save and show cluster result.")
    centroids = kmeans.cluster_centers_  # get centers of clusters
    for (cluster, centroid) in zip(clusters, centroids):  # record and plot one cluster
        cp = ClusterPlotter(category_dic, cluster, centroid, times_rev_dic, times_whatsnew_dic,
                            successive_daysWh, successive_dict)

        print("Start to separate cluster.")
        cp.detail_cluster(CLUSTER_DETAIL_PATH)

        print("Start to time cluster.")
        cp.time_cluster()

        # print("Start to record cluster.")
        # cp.write_to_file(CLUSTER_DATA_DIR, PICKED_CLUSTER_DATA_DIR)

        print("Start to plot cluster.")
        cp.plot_cluster(CLUSTER_IMAGE_DIR)


if __name__ == '__main__':
    cluster_apps(r'D:\programming\workspacePycharm\masterProject\AppCategory')
    '''
    root_dir = r'D:\programming\workspacePycharm\masterProject\Data\Keywords'
    k_path = r'D:\programming\workspacePycharm\masterProject\Data\ChooseK\k.xlsx'

    category = r'Music & Audio'
    app_name = r'com_clearchannel_iheartradio_controller'

    k_df = pandas.read_excel(k_path)
    k = int(k_df[k_df.App == app_name].k)
    cluster_apps([
        (os.path.join(root_dir, 'WhatsNew', app_name), False),
        (os.path.join(root_dir, 'Review', app_name), True)],
        category, app_name, k)
    '''

