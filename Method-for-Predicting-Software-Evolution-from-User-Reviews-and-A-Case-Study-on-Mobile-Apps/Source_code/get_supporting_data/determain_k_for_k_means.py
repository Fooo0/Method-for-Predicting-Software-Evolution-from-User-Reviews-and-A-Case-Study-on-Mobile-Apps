# encoding: utf-8
'''
Created on 2019.01.23

@author: Molly Song
'''


import re
import os
import gc
import numpy
import gensim
import pickle
import traceback
import tracemalloc
from numpy import float32
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans, MiniBatchKMeans
from matplotlib.ticker import MultipleLocator
from main.class_defination import FeatureCase


K_DIR = r'D:\programming\workspacePycharm\masterProject\Data\ChooseK'
KEYWORD_DIR = r'D:\programming\workspacePycharm\MasterProject\Data\Keywords'
MODEL_PATH = r'D:\programming\workspacePycharm\masterProject\Word2VectorModel\wiki.en.text.vector'


def try_k(vector_array, k_candidate, f_name):
    print("Start to test K")
    chi_test_results = []
    # silhouette_coefficient = []
    sse = []
    for try_k in k_candidate:
        print("Start cluster")
        try:
            kmeans = KMeans(n_clusters=try_k, max_iter=400, copy_x=False).fit(
                vector_array)  # use euclidean metric to measure similarity
            # kmeans = MiniBatchKMeans(n_clusters=try_k, max_iter=300, batch_size=80000, n_init=10).fit(
            #     vector_array)
        except BaseException:
            kmeans = KMeans(n_clusters=try_k, max_iter=400, copy_x=False).fit(
                vector_array)  # use euclidean metric to measure similarity
            # kmeans = MiniBatchKMeans(n_clusters=try_k, max_iter=300, batch_size=80000, n_init=10).fit(
            #     vector_array)
        print("Finish cluster with k = %d." % try_k)
        chi = metrics.calinski_harabaz_score(vector_array, kmeans.labels_)  # Calinski-Harabasz Index
        # sil = metrics.silhouette_score(all_vec, kmeans.labels_)  # silhouette coefficient
        sse.append(kmeans.inertia_)
        chi_test_results.append(chi)
        # silhouette_coefficient.append(sil)

    choose_k_file = open(os.path.join(K_DIR, '%s.txt' % f_name), 'w')
    for i in range(len(list(k_candidate))):
        choose_k_file.write('%d %f %f\n' % (k_candidate[i], sse[i], chi_test_results[i]))
    choose_k_file.close()

    fig = plt.figure()
    ax1 = fig.add_subplot(111)

    xmajorLocator = MultipleLocator(10)
    xminorLocator = MultipleLocator(5)
    ax1.xaxis.set_major_locator(xmajorLocator)
    ax1.xaxis.set_minor_locator(xminorLocator)

    ax1.set_xlabel('K')
    ax1.set_ylabel('SSE')
    # ax1.set_xticks(k_candidate)
    ax1.plot(k_candidate, sse, color='#87CEFA')
    plt.xticks(fontsize=6)

    ax2 = ax1.twinx()
    ax2.set_ylabel('Calinski-Harabasz Index')
    ax2.plot(k_candidate, chi_test_results, color='#f36198')

    plt.savefig(os.path.join(K_DIR, '%s.svg'%f_name), format='svg')
    print('SAVE')


# other: 14078339
def get_feature_vectors(catg_path):
    model = gensim.models.KeyedVectors.load_word2vec_format(MODEL_PATH, binary=False)
    model_vectors = model.wv
    del model
    gc.collect()
    all_vec = []
    # all_vector_array = numpy.zeros((4421369, 200), dtype=float32)
    # array_row = 0
    app_count = 0
    vector_count = 0
    # for category in ['Entertainment', 'Games']:
    # for category in ['Books & Reference', 'Business', 'Education',
    #                  'Social', 'Communication', 'Finance', 'Maps & Navigation',
    #                  'News & Magazines', 'Travel & Local']:
    for category in ['Shopping']:
        file_name = '{}.txt'.format(category)
    # for file_name in os.listdir(catg_path):
    #     category = file_name.split('.')[0]
    #     print(category)
    #     if category == 'Games':
    #         continue
        with open(os.path.join(catg_path, file_name), 'r') as f:
            for an in f:
        # f = open(os.path.join(catg_path, file_name), 'r')
        # app_names = f.readlines()
        # f.close()
        # for an in app_names:
                app_count += 1
                an = an.strip('\n')
                print(an)
                source_data_dir = [(os.path.join(KEYWORD_DIR, 'Whatsnew', an), False),
                                   (os.path.join(KEYWORD_DIR, 'Review', an), True)]
                for sdd, flag in source_data_dir:
                    for fileName in sorted(os.listdir(sdd), key=lambda x: int(x.split(".")[0])):
                        time_day = int(fileName.split(".")[0])  # get date
                        print(time_day)
                        # one_file = open(os.path.join(sdd, fileName), 'r',
                        #                encoding='utf-8', errors='ignore')
                        # lines = one_file.readlines()  # feature informations
                        # one_file.close()
                        # freq_check = 0
                        with open(os.path.join(sdd, fileName), 'r',
                                  encoding='utf-8', errors='ignore') as one_file:
                            for line in one_file:
                        # for line in lines:  # one feature information
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
                                vector_count += 1
                print('~~~~ {} App has over ~~~~'.format(app_count))
    # print('vector number: {}'.format(vector_count))
    return numpy.array(all_vec, dtype=float32)
    # return all_vector_array



    catg_path = r'D:\programming\workspacePycharm\masterProject\AppCategory'

    # tracemalloc.start()
    vector_array = get_feature_vectors(catg_path)

    # snapshot = tracemalloc.take_snapshot()
    # top_stats = snapshot.statistics('lineno')
    # print(tracemalloc.get_traced_memory())
    # for stat in top_stats[:20]:
    #     print(stat)

    k_candidate = range(10, 310, 10)
    record_k_file_name = r'Shopping\10_300_10'

    gc.collect()
    # MiniBatchKMeans(n_clusters=20000, max_iter=300, batch_size=80000, n_init=5).fit(vector_array)
    try_k(vector_array, k_candidate, record_k_file_name)

'''
Entertainment, Games: 8160925(20000, 100)
Books & Reference, Business, Education, Social, Communication, Finance, Maps & Navigation, News & Magazines, Travel & Local: 7412227(20000, 100):900
Music & Audio, Photography, Personalization, Productivity, Tools, Weather, Lifestyle, Shopping: 4421369(10000, 100): 800
'''
